"""Train R-JEPA on precomputed representations (phase 1: symmetric encoder).

Wires ``PrecomputedReasoningDataset`` + length-bucketed batch sampling
into the existing ``Trainer``.  No encoder is resident: the forward
consumes stored ctx embeddings and pooled step targets directly, so GPU
memory holds only the predictor.

Causal variant: the collate packs each sample's stream as
``[ctx | steps_0..n-2]`` padded to the batch max TOTAL (no per-axis
bounding-box waste), and the predictor inserts its start token with
per-sample packed RoPE positions - matching the inference regime.
Cross variant: per-axis padded tensors, as the text path produced.

Run (WANDB_MODE=disabled to skip logging):
    python scripts/openr1/openr1_rjepa_precomputed_train.py \
        --max-train-steps 100 --checkpoint-dir /tmp/rjepa_checkpoints
"""

import argparse
import sys
import yaml
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import torch
import wandb

from torch.utils.data import DataLoader, Subset

from src.utils.optim import param_groups_with_decay

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.helper.collate import pad_collate_precomputed_causal, pad_collate_precomputed_cross
from src.data.helper.sampler import BucketedBatchSampler
from src.data.precomputed_dataset import PrecomputedReasoningDataset
from src.models.r_jepa.r_jepa import CausalRJEPA, CrossRJEPA
from src.trainer.r_jepa.r_jepa_trainer import Trainer

from tools.telegram_service import send_bot_message
from tools.on_error import format_error

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument(
    #     "--ctx-dir",
    #     default=str(REPO_ROOT / "data" / "openr1_science_precompute" / "ctx"),
    # )
    # parser.add_argument(
    #     "--targets-dir",
    #     default=str(REPO_ROOT / "data" / "openr1_science_precompute" / "targets"),
    # )
    # parser.add_argument("--hidden-size", type=int, default=2560)
    # parser.add_argument("--cross-attention", action="store_true")
    # parser.add_argument(
    #     "--max-steps", type=int, default=128,
    #     help="cap reasoning steps per sample (p95=87, max=401 in science)",
    # )
    # parser.add_argument("--batch-size", type=int, default=16)
    # parser.add_argument("--grad-accum", type=int, default=1)
    # parser.add_argument("--max-train-steps", type=int, default=10000)
    # parser.add_argument("--learning-rate", type=float, default=1e-3)
    # parser.add_argument("--warmup-steps", type=int, default=1000)
    # parser.add_argument("--predictor-dim", type=int, default=512)
    # parser.add_argument("--num-heads", type=int, default=8)
    # parser.add_argument("--d-ff", type=int, default=1024)
    # parser.add_argument("--num-layers", type=int, default=4)
    # parser.add_argument("--num-workers", type=int, default=4)
    # parser.add_argument(
    #     "--checkpoint-dir", default=str(REPO_ROOT / "checkpoints" / "rjepa"),
    # )
    # parser.add_argument("--wandb-project", default="r-jepa")
    # parser.add_argument("--wandb-name", default=None)
    # parser.add_argument(
    #     "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    # )
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "config" / "openr1_rjepa_precomputed_train.yaml"))
    return parser.parse_args()


def build_val_split(
    dataset, primary, secondary, batch_size, val_per_bucket, n_rewrite=1,
):
    """Group-aware held-out split over precomputed rows.

    Holds out the last ``val_per_bucket`` GROUPS of each length bucket,
    where a group is the n_rewrite rewrite variants of one instruction
    (recovered as ``sample_index // n_rewrite``).  Every row of a held
    group is excluded from training, including sibling variants that
    landed in other length buckets.  With ``n_rewrite == 1`` each row is
    its own group and the behavior is exactly the original per-row
    bucket-tail holdout.

    Returns ``(train_indices, val_indices)`` lists of dataset row positions.
    """
    keys = np.array([int(k) for k in dataset.keys], dtype=np.int64)
    groups = keys // n_rewrite

    sampler = BucketedBatchSampler(
        primary, secondary=secondary, batch_size=batch_size,
    )
    held_groups = set()
    for members in sampler.buckets:
        seen = list(dict.fromkeys(groups[members].tolist()))  # dedupe, keep order
        held_groups.update(seen[-val_per_bucket:])

    held = np.isin(groups, np.fromiter(held_groups, dtype=np.int64))
    val_indices = np.nonzero(held)[0].tolist()
    train_indices = np.nonzero(~held)[0].tolist()
    return train_indices, val_indices


def main(config) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- data: joined branches + length-bucketed batch sampling ----
    # n_rewrite comes from the config when set; the dataset falls back to
    # the precompute's part-*.json sidecar (and to 1, the standard 1:1
    # join) when unset.  The dataset resolves it so the 1:k ctx join and
    # the group holdout below always agree.
    dataset = PrecomputedReasoningDataset(
        ctx_dir=config["ctx_dir"],
        targets_dir=config["targets_dir"],
        hidden_size=config["encoder_hidden_size"],
        max_steps=config["predictor_max_seq_len"],
        max_ctx_len=config["encoder_max_seq_len"],
        n_rewrite=config.get("n_rewrite"),
    )
    n_rewrite = dataset.n_rewrite

    if config["cross_attention"]:
        primary, secondary = dataset.bucket_keys("cross")
        collate_fn = pad_collate_precomputed_cross
    else:
        primary, secondary = dataset.bucket_keys("causal")
        collate_fn = pad_collate_precomputed_causal

    # Hold out the last few GROUPS of each length bucket for router eval.
    # A group = the n_rewrite rewrite variants of one instruction
    # (sample_index // n_rewrite); with n_rewrite == 1 every row is its
    # own group and this degenerates exactly to the old per-row bucket-tail
    # holdout.  All variants of a held group leave training, even where
    # sibling variants landed in other length buckets.
    train_indices, val_indices = build_val_split(
        dataset,
        primary,
        secondary=secondary,
        batch_size=config["batch_size"],
        val_per_bucket=config.get("val_per_bucket", config["batch_size"]),
        n_rewrite=n_rewrite,
    )

    training_dataset = Subset(dataset, train_indices)
    validation_dataset = Subset(dataset, val_indices)

    train_sampler = BucketedBatchSampler(
        primary[train_indices],
        secondary=None if secondary is None else secondary[train_indices],
        batch_size=config["batch_size"],
    )
    val_sampler = BucketedBatchSampler(
        primary[val_indices],
        secondary=None if secondary is None else secondary[val_indices],
        batch_size=config["batch_size"],
    )

    train_loader = DataLoader(
        training_dataset, batch_sampler=train_sampler, collate_fn=collate_fn,
        num_workers=config["num_workers"],
        persistent_workers=True, pin_memory=True, prefetch_factor=config.get("prefetch_factor", 2)
    )

    val_loader = DataLoader(
        validation_dataset, batch_sampler=val_sampler, collate_fn=collate_fn,
        num_workers=config["num_workers"],
        persistent_workers=True, pin_memory=True, prefetch_factor=config.get("prefetch_factor", 2)
    )

    # ---- model: predictor only, no encoder resident ----
    predictor_kwargs = dict(
        encoder_dim=config["encoder_hidden_size"],
        predictor_dim=config["predictor_hidden_size"],
        num_heads=config["predictor_num_heads"],
        d_ff=config["predictor_d_ff"],
        num_layers=config["predictor_num_layers"],
        max_seq_length=config["predictor_max_seq_len"],
        dropout=0.0,
    )
    variant = CrossRJEPA if config["cross_attention"] else CausalRJEPA
    model = variant(
        predictor_kwargs=predictor_kwargs,
        encoder=None,               # precomputed-only: forward never touches an encoder
        encoder_max_seq_length=config["encoder_max_seq_len"],
        encoder_pooling=config["encoder_pooling"],
    ).to(device)

    param_groups = param_groups_with_decay(
        model,
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.1),
        betas=(0.9, 0.95),
        no_decay_patterns=("norm", "bias", "router"),
    )

    optimizer = torch.optim.AdamW(param_groups)

    start_step = 0

    if config.get('from_checkpoint'):
        print(f"Loading weights from checkpoint: {config['from_checkpoint']}")
        checkpoint = torch.load(config['from_checkpoint'], map_location='cpu', weights_only=False)

        model.load_state_dict(checkpoint['model_state_dict'], strict=True)

        if config.get('is_resume', False):
            missing = [k for k in ['optimizer_state_dict', 'step'] if k not in checkpoint]
            if missing:
                raise ValueError(f"is_resume=True but the checkpoint is missing: {missing}")

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            for param_group in optimizer.param_groups:
                # Current config wins over the checkpoint's stored lr/weight_decay.
                param_group['lr'] = config['learning_rate']
                param_group['weight_decay'] = config['weight_decay']

            start_step = checkpoint['step']
            assert isinstance(start_step, int), \
                f"checkpoint 'step' must be an int to resume, got {start_step!r}"

            print(f"is_resume=True: restored optimizer, resuming from step {start_step}.")
        else:
            print("is_resume=False: weights warm-started, fresh optimizer, step 0.")

    wandb.init(
        project=config["project_name"], 
        name=config["run_name"], 
        id=config.get("run_id", None),
        resume="must" if config.get('run_id') else None,
        dir=config["wandb_dir"],
        config=config
    )

    trainer = Trainer(model, train_loader, val_loader, optimizer, config, device, start_step=start_step)
    t_start = time.time()
    final_step = trainer.train()

    if config.get("save_final_checkpoint", True) and final_step % config.get("save_every", 5000) != 0:
        # Do a final checkpoint save if requested and the last step was not already a save step.
        trainer.save_checkpoint(final_step)

    wandb.finish()

    text = (
        "Training finished: \n"
        f"Project: {config['project_name']}, Run: {config['run_name']}\n"
        f"Ctx: {config['ctx_dir']}, Targets: {config['targets_dir']}\n"
        f"Model variant: {'CrossAttention' if config['cross_attention'] else 'CausalAttention'}\n"
        f"Elapsed: {(time.time() - t_start):.0f}s, Steps: {final_step}\n"
        f"Checkpoint dir: {config['checkpoint_dir']}\n"
    )
    send_bot_message(text)

if __name__ == "__main__":
    args = parse_args()    

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    try:
        main(config)
    except Exception as e:
        err_msg = format_error(e)
        text = (
            "Training failed:\n"
            f"Project: {config.get('project_name', '?')}, Run: {config.get('run_name', '?')}\n"
            f"Ctx: {config.get('ctx_dir', '?')}, Targets: {config.get('targets_dir', '?')}\n"
            f"Model variant: {'CrossAttention' if config.get('cross_attention') else 'CausalAttention'}\n"
            f"Details:\n{err_msg}"
        )
        send_bot_message(text)
        raise
