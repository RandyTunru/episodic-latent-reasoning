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

        rewrite_parts = sorted(
            str(p) for p in Path(config["rewrite_dir"]).glob("part-*.parquet")
        )
        rw = pq.read_table(
            rewrite_parts, columns=["orig_row_id", "cot_variant", "messages",
                                    "cot_text"],
        ).to_pandas()

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

    Layout per sample: [instruction | latents-or-cot | answer]; padded
    positions carry attention_mask=0 and label=-100.
    """
    max_ctx = config["max_ctx_tokens"]
    max_ans = config["max_answer_tokens"]
    max_lat = config.get("max_latent_steps", 128)
    max_cot = config.get("max_cot_tokens", 1024)

    inst_ids, ans_ids, latents, cot_ids = [], [], [], []
    for inst, ans, cot, lat in rows:
        inst_ids.append(tokenizer(
            inst, truncation=True, max_length=max_ctx,
            add_special_tokens=True,
        )["input_ids"])
        ans_ids.append(tokenizer(
            ans, truncation=True, max_length=max_ans,
            add_special_tokens=False,
        )["input_ids"] + [tokenizer.eos_token_id])
        if mode == "latents":
            latents.append(lat[:max_lat])
        elif mode == "cot":
            cot_ids.append(tokenizer(
                cot, truncation=True, max_length=max_cot,
                add_special_tokens=False,
            )["input_ids"])

    B = len(rows)
    L_inst = max(len(x) for x in inst_ids)
    L_ans = max(len(x) for x in ans_ids)
    S = max(l.size(0) for l in latents) if mode == "latents" else 0
    L_cot = max(len(x) for x in cot_ids) if mode == "cot" else 0
    total = L_inst + S + L_cot + L_ans
    H = embed_fn.weight.size(1)  # (vocab, hidden) - size(1) is the hidden dim

    embeds = torch.zeros(B, total, H, dtype=torch.bfloat16, device=device)
    labels = torch.full((B, total), -100, dtype=torch.long, device=device)
    mask = torch.zeros(B, total, dtype=torch.long, device=device)

    for i in range(B):
        p = 0
        with torch.no_grad():  # frozen token embeddings only
            e_inst = embed_fn(torch.tensor(inst_ids[i], device=device))
        embeds[i, :e_inst.size(0)] = e_inst

        mask[i, :len(inst_ids[i])] = 1
        p += L_inst
        if mode == "latents":
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
        config["model_id"], dtype=torch.bfloat16,
    ).to(device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    embed_fn = model.get_input_embeddings()

    proj = nn.Linear(
        config["encoder_hidden_size"], model.config.hidden_size,
    ).to(device)

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
