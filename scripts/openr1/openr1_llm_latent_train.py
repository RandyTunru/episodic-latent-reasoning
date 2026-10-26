"""Fine-tune an LLM to answer from instruction + PREDICTOR-GENERATED latent
reasoning (the pilot's Step B).

Pilot scope: the LLM is FROZEN and only a latent projection is trained -
a learned map from the latent space (Qwen3-Embedding-4B, the targets
encoder) into the LLM's hidden space, injected as soft prompts between
the instruction and the answer.  This isolates the question the pilot
asks: can the LLM READ the latents at all?  (LoRA on the LLM is a
follow-up once this signal is positive.)

Input conditions, selected by config ``mode``:
    latents - [instruction | proj(latents)] -> answer   (the system)
    cot     - [instruction | rewritten CoT text] -> answer  (dense-CoT baseline)
    none    - [instruction] -> answer                  (instruction-only baseline)

Data: rewritten-cot parquet (instruction + answer) joined with the
latent precompute branch on sample_index = orig_row_id * n_rewrite +
cot_variant.  The train/val split is the SAME bucket-tail holdout the
predictor used (build_val_split), so the end-to-end eval rows are never
trained on.

Loss: causal LM on answer tokens only (instruction/latent positions are
-100 masked).  Only the projection checkpoint is saved; the frozen LLM
is reloaded from the hub each run.
"""
from dotenv import load_dotenv
load_dotenv() 

import argparse
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn as nn
import wandb
import yaml
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.openr1.openr1_rjepa_precomputed_train import build_val_split
from src.data.precomputed_dataset import PrecomputedReasoningDataset
from tools.telegram_service import send_bot_message
from tools.on_error import format_error


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    return p.parse_args()


def extract_answer(response: str) -> str:
    """The final answer is whatever follows the LAST closing think tag."""
    return response.rsplit("</think>", 1)[-1].strip()


class LatentAnswerDataset(Dataset):
    """Rows: (instruction, answer, cot_text, latents-or-None).

    Joined on sample_index = orig_row_id * n_rewrite + cot_variant from
    the rewritten-cot parquet and the latent precompute branch.  ``keep``
    (a set of sample_index) filters to the predictor's train/val rows.
    """

    def __init__(self, config, latents_dir=None, mode="latents", n_rewrite=1,
                 hidden_size=2560, keep=None):
        self.mode = mode
        self.hidden_size = hidden_size

        lat_by_key = {}
        if mode == "latents":
            lat_parts = sorted(
                str(p) for p in Path(latents_dir).glob("part-*.parquet")
            )
            lat = pq.read_table(
                lat_parts, columns=["sample_index", "step_latents"],
            ).to_pandas()
            lat_by_key = dict(zip(lat["sample_index"].tolist(),
                                  lat["step_latents"].tolist()))

        if config.get("data_format") == "gsm8k":
            # GSM8K rows: {question, answer} from HF, joined with the
            # latent precompute on sample_index == row index.  The answer
            # IS the full step-by-step solution (ends "#### N") - the
            # same text the LLM must generate at eval.
            from datasets import load_dataset as _load_ds
            gsm = _load_ds(config.get("dataset_id", "openai/gsm8k"), "main",
                           split=config.get("split", "train"))
            self.rows = []
            skipped = 0
            for i, ex in enumerate(gsm):
                if keep is not None and i not in keep:
                    continue
                latents = None
                if mode == "latents":
                    blob = lat_by_key.get(i)
                    if blob is None:
                        skipped += 1
                        continue
                    latents = torch.frombuffer(
                        blob, dtype=torch.bfloat16,
                    ).reshape(-1, hidden_size).float()
                self.rows.append((ex["question"], ex["answer"], "", latents))
            print(
                f"LatentAnswerDataset (gsm8k): {len(self.rows)} rows "
                f"(mode={mode}, skipped {skipped})", flush=True,
            )
            return

        rewrite_parts = sorted(
            str(p) for p in Path(config["rewrite_dir"]).glob("part-*.parquet")
        )
        rw = pq.read_table(
            rewrite_parts, columns=["orig_row_id", "cot_variant", "messages",
                                    "cot_text"],
        ).to_pandas()

        self.rows = []
        skipped = 0
        for r in rw.itertuples(index=False):
            key = r.orig_row_id * n_rewrite + r.cot_variant
            if keep is not None and key not in keep:
                continue
            messages = r.messages
            instruction = messages[0]["content"]
            answer = extract_answer(messages[1]["content"])
            if not answer:
                skipped += 1
                continue
            latents = None
            if mode == "latents":
                blob = lat_by_key.get(key)
                if blob is None:
                    skipped += 1
                    continue
                latents = torch.frombuffer(
                    blob, dtype=torch.bfloat16,
                ).reshape(-1, hidden_size).float()
            self.rows.append((instruction, answer, r.cot_text, latents))
        print(
            f"LatentAnswerDataset: {len(self.rows)} rows (mode={mode}, "
            f"skipped {skipped})", flush=True,
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def build_batch(rows, tokenizer, embed_fn, proj, config, device, mode):
    """Assemble one batch as inputs_embeds/labels/attention_mask.

    Layout per sample (chat template + think-block boundaries):

        [<|im_start|>user ... <think>] [latents-or-cot] [</think>...] [answer]

    The template is rendered with ``enable_thinking=False``, which
    pre-fills an EMPTY ``<think>\n\n</think>`` block - the Qwen3
    non-thinking mechanism.  We inject the reasoning payload INSIDE that
    block: projected latents (mode latents) or CoT text tokens (mode
    cot); mode none leaves the block empty.  The model's own prior does
    the work: content inside <think> is reasoning to read, and a
    completed think block means "answer now".  Padded positions carry
    attention_mask=0 and label=-100; only answer tokens carry labels.
    """
    max_ctx = config["max_ctx_tokens"]
    max_ans = config["max_answer_tokens"]
    max_lat = config.get("max_latent_steps", 128)
    max_cot = config.get("max_cot_tokens", 1024)

    think_id = tokenizer.convert_tokens_to_ids("<think>")
    close_id = tokenizer.convert_tokens_to_ids("</think>")
    if think_id is None or close_id is None:
        raise RuntimeError("tokenizer has no <think>/</think> tokens - "
                           "cannot build think-block boundaries")

    pre_ids, post_ids, ans_ids, latents, cot_ids = [], [], [], [], []
    for inst, ans, cot, lat in rows:
        # truncate the instruction text first (tokenize -> decode is
        # lossless for text without special tokens), then template it
        trunc = tokenizer(
            inst, truncation=True, max_length=max_ctx,
            add_special_tokens=False,
        )["input_ids"]
        inst_text = tokenizer.decode(trunc, skip_special_tokens=True)
        tpl = tokenizer.apply_chat_template(
            [{"role": "user", "content": inst_text}],
            add_generation_prompt=True, enable_thinking=False,
            return_tensors="pt",
        )["input_ids"][0].tolist()
        open_pos = [i for i, t in enumerate(tpl) if t == think_id]
        close_pos = [i for i, t in enumerate(tpl) if t == close_id]
        if not open_pos or not close_pos:
            raise RuntimeError(
                "chat template produced no <think> block (template "
                "version may differ; expected the empty think block)"
            )
        pre_ids.append(tpl[:open_pos[0] + 1])   # ...up to and incl. <think>
        post_ids.append(tpl[close_pos[0]:])     # </think>... onward
        # EOS is appended ONLY when the answer fits inside the cap.
        # A truncated answer is left UNENDED: appending EOS at the cut
        # teaches the projection false endings (the 513-token self-cut
        # bug).  An unended prefix gives honest partial supervision.
        a_full = tokenizer(
            ans, truncation=False, add_special_tokens=False,
        )["input_ids"]
        if len(a_full) <= max_ans:
            ans_ids.append(a_full + [tokenizer.eos_token_id])
        else:
            ans_ids.append(a_full[:max_ans])
        if mode == "latents":
            latents.append(lat[:max_lat])
        elif mode == "cot":
            cot_ids.append(tokenizer(
                cot, truncation=True, max_length=max_cot,
                add_special_tokens=False,
            )["input_ids"])

    B = len(rows)
    L_pre = max(len(x) for x in pre_ids)
    L_post = max(len(x) for x in post_ids)
    L_ans = max(len(x) for x in ans_ids)
    S = max(l.size(0) for l in latents) if mode == "latents" else 0
    L_cot = max(len(x) for x in cot_ids) if mode == "cot" else 0
    total = L_pre + S + L_cot + L_post + L_ans
    H = embed_fn.weight.size(1)  # (vocab, hidden) - size(1) is the hidden dim

    embeds = torch.zeros(B, total, H, dtype=torch.bfloat16, device=device)
    labels = torch.full((B, total), -100, dtype=torch.long, device=device)
    mask = torch.zeros(B, total, dtype=torch.long, device=device)

    for i in range(B):
        p = 0
        with torch.no_grad():  # frozen token embeddings only
            e_pre = embed_fn(torch.tensor(pre_ids[i], device=device))
            e_post = embed_fn(torch.tensor(post_ids[i], device=device))
        embeds[i, :e_pre.size(0)] = e_pre
        mask[i, :len(pre_ids[i])] = 1
        p += L_pre
        if mode == "latents":
            # proj forward is OUTSIDE no_grad: its output must carry the
            # autograd graph so the projection receives gradients.
            lat_i = latents[i].to(device, dtype=torch.bfloat16)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                embeds[i, p:p + lat_i.size(0)] = proj(lat_i)
            mask[i, p:p + latents[i].size(0)] = 1
            p += S
        elif mode == "cot":
            with torch.no_grad():
                e_cot = embed_fn(torch.tensor(cot_ids[i], device=device))
            embeds[i, p:p + e_cot.size(0)] = e_cot
            mask[i, p:p + len(cot_ids[i])] = 1
            p += L_cot
        embeds[i, p:p + e_post.size(0)] = e_post
        mask[i, p:p + len(post_ids[i])] = 1
        p += L_post
        a = ans_ids[i]
        with torch.no_grad():
            e_ans = embed_fn(torch.tensor(a, device=device))
        embeds[i, p:p + e_ans.size(0)] = e_ans
        labels[i, p:p + len(a)] = torch.tensor(a, device=device)
        mask[i, p:p + len(a)] = 1
    return embeds, labels, mask


def main(config_path: str) -> None:
    config = yaml.safe_load(open(config_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = config.get("mode", "latents")

    tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"], dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    for p in model.parameters():
        p.requires_grad = False

    model.gradient_checkpointing_enable() # Enable gradient checkpointing for memory efficiency
    model.eval()
    embed_fn = model.get_input_embeddings()

    proj = nn.Linear(
        config["encoder_hidden_size"], model.config.hidden_size,
    ).to(device)
    if config.get("from_checkpoint"):
        ck = torch.load(config["from_checkpoint"], map_location="cpu",
                        weights_only=False)
        proj.load_state_dict(ck["projection"])
        print(f"projection warm-started from {config['from_checkpoint']}",
              flush=True)

    # ---- the predictor's exact train/val split, mapped to sample_index ----
    pds = PrecomputedReasoningDataset(
        ctx_dir=config["ctx_dir"], targets_dir=config["targets_dir"],
        hidden_size=config["encoder_hidden_size"],
        max_steps=config["predictor_max_seq_len"],
        max_ctx_len=config["encoder_max_seq_len"],
        validate=False,
    )
    primary = pds.bucket_keys("causal")[0]
    n_rewrite = config.get("n_rewrite") or 1
    train_idx, val_idx = build_val_split(
        pds, primary, None, config["batch_size"],
        config.get("val_per_bucket", 16), n_rewrite,
    )
    train_keys = {pds.keys[i] for i in train_idx}
    val_keys = {pds.keys[i] for i in val_idx}

    ds = LatentAnswerDataset(
        config, latents_dir=config.get("latents_dir"), mode=mode,
        n_rewrite=n_rewrite, hidden_size=config["encoder_hidden_size"],
        keep=train_keys | val_keys,
    )

    opt = torch.optim.AdamW(
        proj.parameters(), lr=config["learning_rate"],
        betas=(0.9, 0.95), weight_decay=config.get("weight_decay", 0.01),
    )

    wandb.init(project=config["project_name"], name=config.get("run_name"),
               dir=config.get("wandb_dir", "logs"), config=config)

    batch_size = config["batch_size"]
    grad_accum = config.get("gradient_accumulation_steps", 1)
    max_steps = config["max_steps"]
    step, t0 = 0, time.time()

    def forward_batch(rows):
        embeds, labels, mask = build_batch(
            rows, tokenizer, embed_fn, proj, config, device, mode,
        )
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            out = model(inputs_embeds=embeds, attention_mask=mask, labels=labels)
        return out.loss / grad_accum

    data_iter = iter(DataLoader(ds, batch_size=batch_size, shuffle=True,
                                num_workers=0, collate_fn=lambda x: x))
    print("Starting LLM latent training loop ...", flush=True)
    while step < max_steps:
        opt.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(DataLoader(ds, batch_size=batch_size,
                                            shuffle=True, num_workers=0,
                                            collate_fn=lambda x: x))
                batch = next(data_iter)
            loss = forward_batch(batch)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(proj.parameters(), max_norm=1.0)
        opt.step()
        step += 1

        if step % config.get("log_every", 20) == 0:
            print(f"Step {step:06d} | loss={loss.item():.4f} | "
                  f"{time.time() - t0:.0f}s", flush=True)
            wandb.log({"train/loss": loss.item(), "step": step})

        if step % config.get("save_every", 500) == 0:
            ckpt_dir = Path(config["checkpoint_dir"])
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"projection": proj.state_dict(), "step": step},
                ckpt_dir / f"projection_step_{step:06d}.pt",
            )

    wandb.finish()
    print("done", flush=True)


if __name__ == "__main__":
    args = parse_args()
    try:
        main(args.config)
        send_bot_message(f"LLM latent training finished:\n{args.config}")
    except Exception as e:
        err_msg = format_error(e)
        send_bot_message(f"LLM latent training failed:\n{err_msg}")
        raise
