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
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.helper.collate import pad_collate_precomputed_causal, pad_collate_precomputed_cross
from src.data.helper.sampler import BucketedBatchSampler
from src.data.precomputed_dataset import PrecomputedReasoningDataset
from src.models.r_jepa.r_jepa import CausalRJEPA, CrossRJEPA
from src.trainer.r_jepa.r_jepa_trainer import Trainer

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


def main() -> None:
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- data: joined branches + length-bucketed batch sampling ----
    dataset = PrecomputedReasoningDataset(
        ctx_dir=config["ctx_dir"],
        targets_dir=config["targets_dir"],
        hidden_size=config["encoder_hidden_size"],
        max_steps=config["predictor_max_seq_len"],
        max_ctx_len=config["encoder_max_seq_len"],
    )

    if config["cross_attention"]:
        primary, secondary = dataset.bucket_keys("cross")
        collate_fn = pad_collate_precomputed_cross
    else:
        primary, secondary = dataset.bucket_keys("causal")
        collate_fn = pad_collate_precomputed_causal

    # Hold out the last few rows of each length bucket for router eval:
    # stratified by chain length so the halting metrics see the full
    # length distribution instead of the mostly-short mode.  The holdout
    # is removed from training, and each split gets its own bucketed
    # sampler over ITS OWN coordinate space - bucket membership does not
    # survive subsetting.
    full_sampler = BucketedBatchSampler(
        primary, secondary=secondary, batch_size=config["batch_size"],
    )
    val_samples = full_sampler._sample_per_bucket(
        num_samples=config.get("val_per_bucket", config["batch_size"])
    )
    val_indices = sorted({int(i) for i in val_samples})
    train_indices = np.setdiff1d(
        np.arange(len(dataset)), np.asarray(val_indices), assume_unique=True,
    ).tolist()

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
    )

    val_loader = DataLoader(
        validation_dataset, batch_sampler=val_sampler, collate_fn=collate_fn,
        num_workers=config["num_workers"],
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

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(), lr=config["learning_rate"],
        betas=(0.9, 0.95), weight_decay=config.get("weight_decay", 0.1),
    )

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
    final_step = trainer.train()

    if config.get("save_final_checkpoint", True) and final_step % config.get("save_every", 5000) != 0:
        # Do a final checkpoint save if requested and the last step was not already a save step.
        trainer.save_checkpoint(final_step)

    wandb.finish()

if __name__ == "__main__":
    main()
