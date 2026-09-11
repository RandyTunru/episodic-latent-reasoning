"""R-JEPA trainer with masked smooth-L1 regression and router BCE loss.

Follows the same structure as the I-JEPA trainers in ``/home/analta/work/jepa-exp/``:
infinite-dataloader iterator, AMP, gradient accumulation, cosine LR schedule
with linear warmup, and periodic checkpointing.

The loss has two components:

1. **Smooth L1** - regress predicted latent states onto frozen-encoder target
   representations (mean-pooled per reasoning step). Masked so that padding
   step positions do not contribute.  
2. **BCE (router)** - binary cross-entropy on the halting router.
   Pseudo-labels are derived from the position structure: all real
   intermediate steps are "continue" (1), the last real step is "halt" (0),
   and padding positions are ignored via the loss mask.
"""

import math
import os
from typing import Optional

import torch
import torch.nn.functional as F
import wandb

from src.utils.metrics import ThroughputMonitor


class Trainer:
    def __init__(
        self,
        model,
        dataloader,
        optimizer,
        config: dict,
        device: torch.device,
        start_step: int = 0,
    ):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.start_step = start_step

        self.grad_accum_steps = config.get("gradient_accumulation_steps", 1)
        self.router_alpha = config.get("router_alpha", 0.1)

        self.checkpoint_dir = config["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Throughput estimation - context + steps lengths vary, so the
        # monitor uses encoder_max_seq_length as a rough upper bound.
        self.monitor = ThroughputMonitor(
            config["batch_size"] * self.grad_accum_steps,
            config["encoder_max_seq_len"] + config["predictor_max_seq_len"],
        )

    # ------------------------------------------------------------------
    # Learning rate schedule
    # ------------------------------------------------------------------

    def get_lr(self, step: int) -> float:
        """Cosine learning-rate schedule with linear warmup."""
        max_lr = self.config["learning_rate"]
        min_lr = max_lr * 0.1
        warmup_steps = self.config.get("warmup_steps", 1000)
        total_steps = self.config["max_steps"]

        if step < warmup_steps:
            return max_lr * (step / max(warmup_steps, 1))

        if step > total_steps:
            return min_lr

        decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

    def train(self) -> int:
        self.model.train()

        step = self.start_step
        use_amp = self.device.type == "cuda"
        ctx = torch.autocast(
            device_type=self.device.type, dtype=torch.bfloat16, enabled=use_amp,
        )

        data_iter = iter(self.dataloader)

        print("Starting R-JEPA training loop …")
        self.monitor.start()

        while step < self.config["max_steps"]:
            # ---- LR schedule ----
            lr = self.get_lr(step)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            # ---- gradient accumulation ----
            for _ in range(self.grad_accum_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    if self.config['no_epochs']:
                        print("Dataset exhausted. Stopping training.")
                        return step
                    data_iter = iter(self.dataloader)
                    batch = next(data_iter)

                # Move to device and forward.  The model branches on the
                # batch dict's contents (text path vs precomputed path).
                batch = {k: v.to(self.device) for k, v in batch.items()}

                with ctx:
                    preds, targets, router_logits, step_valid = self.model(batch)
                    loss, regression_loss, router_bce = self._compute_loss(
                        preds, targets, router_logits, step_valid,
                    )
                    loss = loss / self.grad_accum_steps

                loss.backward()
                accum_loss += loss.item()

            # ---- optimiser step ----
            torch.nn.utils.clip_grad_norm_(
                self.model.trainable_parameters(), max_norm=1.0,
            )
            self.optimizer.step()

            # ---- logging ----
            if step % self.config.get("log_every", 50) == 0:
                tps = self.monitor.get_tps()
                wandb.log({
                    "train/loss": accum_loss,
                    "train/regression_loss": regression_loss,
                    "train/router_bce": router_bce,
                    "train/learning_rate": lr,
                    "metrics/throughput_tps": tps,
                    "step": step,
                })
                print(
                    f"Step {step:06d} | loss={accum_loss:.4f} | "
                    f"reg_loss={regression_loss:.4f} | router_bce={router_bce:.4f} | "
                    f"lr={lr:.2e} | tps={tps:.0f}"
                )

            # ---- checkpointing ----
            if step > 0 and step % self.config.get("save_every", 5000) == 0:
                self.save_checkpoint(step)

            step += 1

        return step

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def _compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        router_logits: torch.Tensor,
        step_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Joint MSE + router BCE loss, masked for padding.

        The loss has two components:

        **Smooth L1** - Each predicted latent state should match the frozen
        encoder's representation of the corresponding reasoning step.  This
        is the core JEPA objective: learn to forecast the next reasoning
        state in latent space.  Smooth L1 (β=1.0) is used instead of MSE
        because it is less sensitive to outliers - large prediction errors
        on early reasoning steps won't dominate the gradient.  Only valid
        (non-padding) step positions contribute.

        **Router BCE** - The router learns when to halt.  Pseudo-labels are
        derived from the position structure:
          - All intermediate valid steps → label 1 (continue reasoning)
          - The last valid step → label 0 (halt here)
          - Padding positions → excluded from loss

        The hyperparameter α (router_alpha) balances the two objectives.
        Typical values are 0.05–0.2, tuned so that the MSE and BCE losses
        are roughly the same order of magnitude at the start of training.

        Args:
            predictions: ``(B, S, E)`` predicted latent states.
            targets: ``(B, S, E)`` target representations (stop-gradient).
            router_logits: ``(B, S)`` raw halting scores.
            step_valid_mask: ``(B, S)`` bool, ``True`` = real step.

        Returns:
            Scalar loss = smooth L1 + α · BCE.
        """
        B, S, _ = predictions.shape
        valid_float = step_valid_mask.float()
        num_valid = valid_float.sum().clamp(min=1)

        # --- Smooth L1 (per-step reconstruction of the next latent state) ---
        reg_per_step = F.smooth_l1_loss(
            predictions, targets, reduction="none", beta=1.0,
        ).mean(dim=-1)  # (B, S)
        regression_loss = (reg_per_step * valid_float).sum() / num_valid

        # --- Router BCE ---
        # Pseudo-labels are derived from position structure - no external supervision needed.
        router_labels = valid_float.clone()  # 1.0 where valid, 0.0 where pad
        for b in range(B):
            pos = step_valid_mask[b].nonzero(as_tuple=True)[0]
            if pos.numel() > 0:
                router_labels[b, pos[-1]] = 0.0  # last valid step → halt

        bce_per_step = F.binary_cross_entropy_with_logits(
            router_logits, router_labels, reduction="none",
        )  # (B, S)
        router_bce = (bce_per_step * valid_float).sum() / num_valid

        return regression_loss + self.router_alpha * router_bce, regression_loss, router_bce

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, step: int, label: Optional[str] = None):
        name = label if label is not None else str(step)
        filepath = os.path.join(self.checkpoint_dir, f"model_step_{name}.pt")
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step": step,
            "config": self.config,
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved → {filepath}")
