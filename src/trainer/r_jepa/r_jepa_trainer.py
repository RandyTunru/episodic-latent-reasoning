"""R-JEPA trainer with masked smooth-L1 regression and router BCE loss.

Follows the same structure as the I-JEPA trainers in ``/home/analta/work/jepa-exp/``:
infinite-dataloader iterator, AMP, gradient accumulation, cosine LR schedule
with linear warmup, and periodic checkpointing.

The loss has two components:

1. **Smooth L1** - regress predicted latent states onto frozen-encoder target
   representations (mean-pooled per reasoning step). Masked so that padding
   step positions do not contribute.  
2. **BCE (router)** - binary cross-entropy on the halting router.
   Pseudo-labels are derived from the position structure: the halt
   position is the positive class (label 1) - it sits at the last
   visible step when the true end (``num_steps_true``) is inside the
   row - while all other real steps are "continue" (0); cut rows show
   all continues.  Padding positions are ignored via the loss mask,
   and ``router_halt_weight`` (the BCE ``pos_weight``) optionally
   amplifies the rare halt class.
"""

import math
import os
from typing import Optional

import torch
import torch.nn.functional as F
import wandb

from src.utils.metrics import ThroughputMonitor
from src.trainer.modules.lr_scheduler import LRScheduler


class Trainer:
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        config: dict,
        device: torch.device,
        start_step: int = 0,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.start_step = start_step

        self.grad_accum_steps = config.get("gradient_accumulation_steps", 1)
        self.router_alpha = config.get("router_alpha", 0.1)
        self.eval_every = config.get("eval_every", 5000)

        self.checkpoint_dir = config["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Throughput estimation - context + steps lengths vary, so the
        # monitor uses encoder_max_seq_length as a rough upper bound.
        self.monitor = ThroughputMonitor(
            config["batch_size"] * self.grad_accum_steps,
            config["encoder_max_seq_len"] + config["predictor_max_seq_len"],
        )

    def train(self) -> int:
        self.model.train()

        step = self.start_step
        use_amp = self.device.type == "cuda"
        ctx = torch.autocast(
            device_type=self.device.type, dtype=torch.bfloat16, enabled=use_amp,
        )

        scheduler = LRScheduler(self.config)

        data_iter = iter(self.train_dataloader)

        print("Starting R-JEPA training loop …")
        self.monitor.start()

        while step < self.config["max_steps"]:
            # ---- LR schedule ----
            lr = scheduler.get_lr(step)
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
                        metrics = self._validate(step, ctx)
                        wandb.log({**metrics, "step": step})
                        return step
                    data_iter = iter(self.train_dataloader)
                    batch = next(data_iter)

                # Move to device and forward.  The model branches on the
                # batch dict's contents (text path vs precomputed path).
                batch = {k: v.to(self.device) for k, v in batch.items()}

                with ctx:
                    preds, targets, router_logits, step_valid = self.model(batch)
                    loss, regression_loss, router_bce = self._compute_loss(
                        preds, targets, router_logits, step_valid,
                        num_steps=batch["num_steps"],
                        num_steps_true=batch.get("num_steps_true"),
                    )
                    loss = loss / self.grad_accum_steps

                loss.backward()
                accum_loss += loss.item()

            # ---- optimiser step ----
            torch.nn.utils.clip_grad_norm_(
                self.model.trainable_parameters(), max_norm=1.0,
            )
            self.optimizer.step()

            # ---- validation ----
            metrics = {}
            if step % self.eval_every == 0:
                metrics = self._validate(step, ctx)

            # ---- logging ----
            if step % self.config.get("log_every", 50) == 0:
                tps = self.monitor.get_tps()
                wandb.log({
                    **metrics, # log validation metrics here instead of at the end of _validate() to avoid double logging
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

        # Final validation if the cadence missed the last step.
        # step - 1 is the last step that was actually trained, so we validate on that.
        # using step here would make the validation metric skip the last step if max_steps is not a multiple of eval_every.
        # as it would be 1 short during the last iteration of the loop, and then incremented to max_steps, which would skip the validation here.
        if step > 0 and (step - 1) % self.eval_every != 0:
            metrics = self._validate(step - 1, ctx)
            wandb.log({
                **metrics, 
                "train/loss": accum_loss,
                "train/regression_loss": regression_loss,
                "train/router_bce": router_bce,
                "train/learning_rate": lr,
                "metrics/throughput_tps": tps,
                "step": step - 1
            })

        return step

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self, step: int, ctx) -> dict:
        """Teacher-forced evaluation on the held-out split.

        Runs the same forward as training  
        The model is run in evaluation mode, and the router logits are used to determine the halting positions.
        Halting is read off the router logits the way ``infer()`` does: the first valid position whose logit is > 0
        (halt is the positive class).

        Logged under ``val/``:

        * ``loss`` / ``regression_loss`` / ``router_bce`` - eval-set loss
          components (masked means over all valid positions).
        * ``halt_rate`` - of the rows whose true end is visible in the
          window (``num_steps_true == num_steps``), the fraction where
          the router halted before the window ended.
        * ``signed_mean_offset`` / ``abs_mean_offset`` / ``median_offset``
          - predicted minus true halt position over visible-end rows that
          halted (negative = early).  The median is robust to the
          no-halt tail; the sign matters more than the magnitude here,
          since early halts truncate reasoning the answer needs while
          late halts only cost cheap predictor steps.
        * ``exact_halt`` / ``within_1`` - fraction of those rows with
          |offset| <= 0 / 1.
        * ``false_halt_rate`` - of CUT rows (true end beyond the window),
          the fraction where the router halted anyway.
        * ``f1`` / ``precision`` / ``recall`` - per-position halt
          classification over all valid positions (the imbalance-safe
          view of the 1-halt-per-chain problem).
        * ``bucket_*`` - halt rate and mean offset split by true chain
          length, to catch length-prior habits.
        """
        self.model.eval()

        logits_list, valid_list, ns_list, ns_true_list = [], [], [], []
        weighted = {"loss": 0.0, "regression_loss": 0.0, "router_bce": 0.0}
        num_valid_total = 0

        with torch.no_grad():
            for val_batch in self.val_dataloader:
                val_batch = {k: v.to(self.device) for k, v in val_batch.items()}

                with ctx:
                    preds, targets, router_logits, step_valid = self.model(val_batch)
                    loss, regression_loss, router_bce = self._compute_loss(
                        preds, targets, router_logits, step_valid,
                        num_steps=val_batch["num_steps"],
                        num_steps_true=val_batch.get("num_steps_true"),
                    )

                n_valid = int(step_valid.sum())
                num_valid_total += n_valid

                # _compute_loss returns per-batch masked means; re-weight by
                # each batch's valid-position count so the final mean is a
                # global per-position mean, not a per-batch mean (length
                # bucketing makes per-batch valid counts vary a lot).
                weighted["loss"] += loss.item() * n_valid
                weighted["regression_loss"] += regression_loss.item() * n_valid
                weighted["router_bce"] += router_bce.item() * n_valid

                logits_list.append(router_logits.detach().cpu())
                valid_list.append(step_valid.detach().cpu())
                ns_list.append(val_batch["num_steps"].cpu())
                ns_true = val_batch.get("num_steps_true")
                if ns_true is None:
                    ns_true = val_batch["num_steps"]
                ns_true_list.append(ns_true.cpu())

        self.model.train()

        logits = torch.cat(logits_list)      # (N, S)
        valid = torch.cat(valid_list).bool() # (N, S)
        ns = torch.cat(ns_list)              # (N,)
        ns_true = torch.cat(ns_true_list)    # (N,)

        # Predicted halt: first valid position with logit > 0 (infer's rule).
        masked_logits = logits.masked_fill(~valid, float("-inf"))
        did_halt = (masked_logits > 0).any(dim=1)
        pred_halt = torch.where(
            did_halt, (masked_logits > 0).int().argmax(dim=1), -1,
        )

        # The halt is knowable only when the true end sits inside the
        # visible window; cut rows should show no halt at all.
        halt_visible = (ns > 0) & (ns_true == ns)
        cut = ~halt_visible
        true_halt = ns - 1
        eval_halt = halt_visible & did_halt  # rows feeding the offset stats

        n_visible = int(halt_visible.sum())
        n_halted = int(eval_halt.sum())

        metrics = {
            "val/loss": weighted["loss"] / max(num_valid_total, 1),
            "val/regression_loss": weighted["regression_loss"] / max(num_valid_total, 1),
            "val/router_bce": weighted["router_bce"] / max(num_valid_total, 1),
            "val/samples": int(ns.size(0)),
        }

        if n_visible > 0:
            metrics["val/halt_rate"] = n_halted / n_visible
            if n_halted > 0:
                offset = (pred_halt - true_halt)[eval_halt].float()
                metrics["val/signed_mean_offset"] = offset.mean().item()
                metrics["val/abs_mean_offset"] = offset.abs().mean().item()
                metrics["val/median_offset"] = offset.median().item()
                metrics["val/exact_halt"] = (offset == 0).float().mean().item()
                metrics["val/within_1"] = (offset.abs() <= 1).float().mean().item()

        n_cut = int(cut.sum())
        if n_cut > 0:
            metrics["val/false_halt_rate"] = (cut & did_halt).sum().item() / n_cut

        # Per-position halt classification (halt = positive class).
        halt_pos = torch.zeros_like(valid)
        rows = halt_visible.nonzero(as_tuple=True)[0]
        halt_pos[rows, true_halt[rows]] = True
        pred_pos = valid & (logits > 0)
        tp = int((halt_pos & pred_pos).sum())
        fp = int((~halt_pos & pred_pos).sum())
        fn = int((halt_pos & ~pred_pos).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        metrics["val/f1"] = 2 * precision * recall / max(precision + recall, 1e-12)

        # Split by true chain length: catch length-prior habits.
        for name, lo, hi in (
            ("le25", 0, 25), ("26_50", 26, 50), ("51_100", 51, 100),
            ("101_200", 101, 200), ("gt200", 201, 10**9),
        ):
            sel = (ns_true >= lo) & (ns_true <= hi)
            n_bucket = int(sel.sum())
            if n_bucket == 0:
                continue
            metrics[f"val/bucket_{name}/count"] = n_bucket
            vis_b = halt_visible & sel
            if vis_b.any():
                metrics[f"val/bucket_{name}/halt_rate"] = (
                    (eval_halt & sel).sum().item() / vis_b.sum().item()
                )
                off_b = (pred_halt - true_halt)[eval_halt & sel]
                if off_b.numel():
                    metrics[f"val/bucket_{name}/mean_offset"] = off_b.float().mean().item()

        halt_rate = metrics.get("val/halt_rate", float("nan"))
        abs_off = metrics.get("val/abs_mean_offset", float("nan"))
        print(
            f"Eval   {step:06d} | loss={metrics['val/loss']:.4f} | "
            f"reg={metrics['val/regression_loss']:.4f} | router_bce={metrics['val/router_bce']:.4f} | "
            f"halt_rate={halt_rate:.3f} | abs_off={abs_off:.1f} | f1={metrics['val/f1']:.3f}"
        )
        return metrics

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def _compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        router_logits: torch.Tensor,
        step_valid_mask: torch.Tensor,
        num_steps: torch.Tensor,
        num_steps_true: Optional[torch.Tensor] = None,
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
          - The last visible step → label 1 (halt here) - only when the
            TRUE end is visible (``num_steps_true == num_steps``, i.e.
            neither the storage cap nor the window cut the sequence).
            Cut rows show all continues: the visible steps are a prefix
            of an episode that keeps going, and labeling the cut "halt"
            would teach the router to stop mid-reasoning.
          - All other valid steps → label 0 (continue reasoning)
          - Padding positions → excluded from loss
        ``router_halt_weight`` is passed as the BCE ``pos_weight``, which
        amplifies the rare halt class against the long continue tail.

        The hyperparameter α (router_alpha) balances the two objectives.
        Typical values are 0.05-0.2, tuned so that the MSE and BCE losses
        are roughly the same order of magnitude at the start of training.

        Args:
            predictions: ``(B, S, E)`` predicted latent states.
            targets: ``(B, S, E)`` target representations (stop-gradient).
            router_logits: ``(B, S)`` raw halting scores.
            step_valid_mask: ``(B, S)`` bool, ``True`` = real step.
            num_steps: ``(B,)`` int64 - visible steps per sample
                (post-cap); delimits the valid positions.
            num_steps_true: ``(B,)`` int64 - TRUE step count X, never
                capped.  ``== num_steps`` iff the true end (the halt
                position) is inside the visible row.

        Returns:
            Scalar loss = smooth L1 + α · BCE.
        """
        valid_float = step_valid_mask.float()
        num_valid = valid_float.sum().clamp(min=1)

        # --- Smooth L1 (per-step reconstruction of the next latent state) ---
        reg_per_step = F.smooth_l1_loss(
            predictions, targets, reduction="none", beta=1.0,
        ).mean(dim=-1)  # (B, S)
        regression_loss = (reg_per_step * valid_float).sum() / num_valid

        # --- Router BCE ---
        # Pseudo-labels are derived from position structure - no external
        # supervision needed.  The halt belongs at the TRUE end of the
        # episode, recorded in ``num_steps_true`` (X).  It is placed only
        # when that end is visible in the row: ``num_steps_true ==
        # num_steps`` means neither the storage cap nor the window cut the
        # sequence.  Cut rows show all continues - the visible steps are a
        # prefix of an episode that keeps going, and labeling the cut
        # "halt" would teach the router to stop mid-reasoning.
        if num_steps_true is None:
            num_steps_true = num_steps  # text path: no caps, num_steps IS X
        router_labels = torch.zeros_like(valid_float)  # 0.0 = continue (or pad)
        halt = (num_steps > 0) & (num_steps_true == num_steps)
        router_labels[halt, num_steps[halt] - 1] = 1.0  # last real step → halt

        # ``pos_weight`` amplifies the positive (halt) class - the single
        # halt signal per chain that the long continue tail would
        # otherwise drown (halt:continue ≈ 1:25 to 1:400).
        bce_per_step = F.binary_cross_entropy_with_logits(
            router_logits, router_labels, reduction="none",
            pos_weight=router_logits.new_tensor(
                self.config.get("router_halt_weight", 1.0)
            ),
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
        
        # Atomic save
        tmp_path = filepath + ".tmp"
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, filepath)
        print(f"Checkpoint saved → {filepath}")
