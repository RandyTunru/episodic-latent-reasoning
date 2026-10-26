"""Fine-tune the base LLM on GSM8K train - the primary CoT baseline for
the adaptation column (matched comparison vs. the rjepa adapted row).

Standard R1-style reasoning SFT, single sequence: the model learns to
PRODUCE its own chain of thought (at eval on test questions there is no
CoT to inject).  Target structure:

    <|im_start|>user
    {question + shared format instruction}<|im_end|>
    <|im_start|>assistant
    <think>
    {solution steps}
    </think>

    #### N

The solution's trailing "#### N" becomes the post-think final answer,
mirroring the rjepa row's [latents | </think> answer] structure.  The
instruction text is the SAME shared FORMAT_INSTRUCTION the benchmark
scripts use, so train and eval prompts match.

Loss: causal LM on the assistant turn only (user part masked).  Full
fine-tune (no new deps), gradient checkpointing, bf16 autocast.  A
small dev slice of train is held out for overfitting monitoring.

Output: checkpoint .pt with model_state_dict (loadable by
compute_base_llm.py --checkpoint), wandb run, telegram handler.
"""
from dotenv import load_dotenv
load_dotenv()

import argparse
import sys
import time
from pathlib import Path

import torch
import wandb
import yaml
from datasets import load_dataset
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import build_prompt, extract_hashes_number, FORMAT_INSTRUCTION
from tools.telegram_service import send_bot_message
from tools.on_error import format_error


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    return p.parse_args()


def split_solution(answer_text: str):
    """Split a GSM8K solution into (steps_text, final_hash_line)."""
    if "####" in answer_text:
        steps, final = answer_text.rsplit("####", 1)
        return steps.strip(), "#### " + final.strip()
    return answer_text.strip(), None


def main(config_path: str) -> None:
    config = yaml.safe_load(open(config_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = config.get("seed", 0)
    torch.manual_seed(seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"], torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)
    model.gradient_checkpointing_enable()

    # ---- data: [question + instruction] -> [think(steps) | #### N] ----
    ds = load_dataset(config.get("dataset_id", "openai/gsm8k"), "main",
                      split=config.get("split", "train"))
    rows = []
    for ex in ds:
        steps, final = split_solution(ex["answer"])
        if final is None:
            continue
        assistant = f"<think>\n{steps}\n</think>\n\n{final}"
        rows.append((build_prompt({"question": ex["question"]}), assistant))
    n_val = max(int(len(rows) * config.get("val_fraction", 0.05)), 1)
    val_rows = rows[-n_val:]
    train_rows = rows[:-n_val]
    print(f"train {len(train_rows)} | val {len(val_rows)}", flush=True)

    def tokenize_batch(batch_rows):
        ids_list, labels_list = [], []
        for user_text, assistant_text in batch_rows:
            ids_full = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text},
                 {"role": "assistant", "content": assistant_text}],
                add_generation_prompt=False, enable_thinking=True,
                return_tensors="pt",
            )["input_ids"][0]
            ids_user = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                add_generation_prompt=True, enable_thinking=True,
                return_tensors="pt",
            )["input_ids"][0]
            labels = torch.full_like(ids_full, -100)
            labels[len(ids_user):] = ids_full[len(ids_user):]
            ids_list.append(ids_full)
            labels_list.append(labels)
        L = max(len(x) for x in ids_list)
        ids_b = torch.full((len(ids_list), L), tokenizer.pad_token_id or
                           tokenizer.eos_token_id)
        lab_b = torch.full((len(ids_list), L), -100)
        for i, (ids, lab) in enumerate(zip(ids_list, labels_list)):
            ids_b[i, :len(ids)] = ids
            lab_b[i, :len(ids)] = lab
        return ids_b, lab_b

    batch_size = config["batch_size"]
    grad_accum = config.get("gradient_accumulation_steps", 1)
    total_steps = config["max_steps"]
    lr = config["learning_rate"]
    warmup = config.get("warmup_steps", 0)

    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            betas=(0.9, 0.95),
                            weight_decay=config.get("weight_decay", 0.01))

    def lr_at(step):
        if step < warmup:
            return lr * step / max(warmup, 1)
        return lr

    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_rows, batch_size=batch_size,
                              shuffle=True, generator=gen,
                              collate_fn=tokenize_batch)
    val_loader = DataLoader(val_rows, batch_size=batch_size, shuffle=False,
                            collate_fn=tokenize_batch)

    wandb.init(project=config["project_name"], name=config.get("run_name"),
               dir=config.get("wandb_dir", "logs"), config=config)

    step, t0 = 0, time.time()
    data_iter = iter(train_loader)
    while step < total_steps:
        opt.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            try:
                ids_b, lab_b = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                ids_b, lab_b = next(data_iter)
            ids_b, lab_b = ids_b.to(device), lab_b.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(input_ids=ids_b, labels=lab_b)
            (out.loss / grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step)
        opt.step()
        step += 1

        if step % config.get("log_every", 20) == 0:
            wandb.log({"train/loss": out.loss.item(),
                       "lr": lr_at(step), "step": step})
            print(f"Step {step:05d} | loss={out.loss.item():.4f} | "
                  f"lr={lr_at(step):.2e} | {time.time() - t0:.0f}s",
                  flush=True)
        if step % config.get("eval_every", 100) == 0:
            model.eval()
            vloss, vn = 0.0, 0
            with torch.no_grad():
                for ids_b, lab_b in val_loader:
                    ids_b, lab_b = ids_b.to(device), lab_b.to(device)
                    with torch.autocast(device_type="cuda",
                                        dtype=torch.bfloat16):
                        out = model(input_ids=ids_b, labels=lab_b)
                    vloss += out.loss.item() * ids_b.size(0)
                    vn += ids_b.size(0)
            wandb.log({"val/loss": vloss / vn, "step": step})
            print(f"Eval {step:05d} | val_loss={vloss / vn:.4f}", flush=True)
            model.train()
        if step % config.get("save_every", 200) == 0:
            ckpt_dir = Path(config["checkpoint_dir"])
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(),
                        "step": step},
                       ckpt_dir / f"model_step_{step:05d}.pt")

    wandb.finish()
    print("done", flush=True)


if __name__ == "__main__":
    args = parse_args()
    try:
        main(args.config)
        send_bot_message(f"GSM8K LLM SFT finished:\n{args.config}")
    except Exception as e:
        err_msg = format_error(e)
        send_bot_message(f"GSM8K LLM SFT failed:\n{err_msg}")
        raise
