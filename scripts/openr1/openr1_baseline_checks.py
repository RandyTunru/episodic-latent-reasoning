"""Data-only baselines for the openr1_all R-JEPA run (h485j1cb).

No checkpoint needed: every baseline below is computed from the
precomputed data itself.  Replicates the trainer's val split exactly
(bucket-tail holdout, n_rewrite=1) and evaluates on those rows:

1. Constant predictors (mean / median target embedding) - the
   no-input floor for the smooth-L1 regression loss.
2. Identity predictor (next step = previous step) - the temporal
   smoothness floor.
3. Length-prior router - a content-free halting policy using only the
   empirical hazard P(chain ends at k | chain reached k), evaluated
   under the trainer's exact metrics (weighted BCE + first-positive
   halt rule, plus an argmax rule for its best case).

Compare against the running rewritten-data jobs (6ha0hxm2 w=7,
kcqobso3 w=5); pass --config to choose the data.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

sys.path.insert(0, "/nvme/analta/work/episodic-latent-reasoning")

from src.data.precomputed_dataset import PrecomputedReasoningDataset
from src.data.helper.sampler import BucketedBatchSampler

BASE = ""
CTX_DIR = f"/data/episodic-latent-reasoning/data/processed/openr1_all_precomputed/ctx/Qwen_Qwen3-4B"
TGT_DIR = f"/data/episodic-latent-reasoning/data/processed/openr1_all_rewritten_precomputed/targets/Qwen_Qwen3-Embedding-4B"
VAL_PER_BUCKET = 16
POS_WEIGHT = float(os.environ.get("BASELINE_POS_WEIGHT", "5"))

# ---- val split, replicated from build_val_split with n_rewrite=1 ----
print("building dataset (eager columns only) ...", flush=True)
ds = PrecomputedReasoningDataset(
    ctx_dir=CTX_DIR, targets_dir=TGT_DIR, hidden_size=2560,
    max_steps=512, max_ctx_len=2048, validate=False,
)
primary = ds.bucket_keys("causal")[0]
sampler = BucketedBatchSampler(primary, secondary=None, batch_size=8)
held = np.concatenate([m[-VAL_PER_BUCKET:] for m in sampler.buckets])
val_idx = np.sort(held)
train_idx = np.setdiff1d(np.arange(len(ds)), val_idx, assume_unique=True)
print(f"rows={len(ds)}  val={len(val_idx)}  train={len(train_idx)}", flush=True)

tgt_paths = sorted(str(p) for p in Path(TGT_DIR).glob("part-*.parquet"))
tgt_cols = pq.read_table(tgt_paths, columns=["sample_index", "num_steps"]).to_pandas()
X_by_key = dict(zip(tgt_cols["sample_index"].tolist(), tgt_cols["num_steps"].tolist()))
X = np.array([X_by_key[k] for k in ds.keys], dtype=np.int64)
eff = ds.num_steps_arr            # effective min(X, P, 512)
visible = X == eff                # true end inside the stored window

print("fetching val rows ...", flush=True)
val_rows = [ds[int(i)] for i in val_idx]
S_lens = [r["num_steps"] for r in val_rows]
n_visible_val = sum(1 for r in val_rows if r["num_steps_true"] == r["num_steps"])
print(f"val rows={len(val_rows)}  visible-end rows={n_visible_val}  "
      f"steps: min={min(S_lens)} max={max(S_lens)} mean={np.mean(S_lens):.0f}", flush=True)

# ---- 1. constant predictors ----
all_tgts = torch.cat([r["step_targets"] for r in val_rows], dim=0)
const_mean = all_tgts.mean(0)
const_median = all_tgts.median(0).values

def const_loss(c, start=0):
    tot, n = 0.0, 0
    for r in val_rows:
        t = r["step_targets"]
        per = F.smooth_l1_loss(c.expand(t.size(0), -1), t, reduction="none", beta=1.0).mean(-1)
        tot += per[start:].sum().item()
        n += t.size(0) - start
    return tot / n

# ---- 2. identity predictor ----
def identity_loss(start=0):
    tot, n = 0.0, 0
    for r in val_rows:
        t = r["step_targets"]
        # step 0 prediction: last ctx token embedding (different encoder
        # space than the step targets - caveat); steps i>=1: previous step.
        pred = torch.cat([r["ctx_embeddings"][-1:], t[:-1]], dim=0)
        per = F.smooth_l1_loss(pred, t, reduction="none", beta=1.0).mean(-1)
        tot += per[start:].sum().item()
        n += t.size(0) - start
    return tot / n

# ---- 3. length-prior router ----
def hazard(rows_idx):
    e = eff[rows_idx]
    v = visible[rows_idx]
    d = np.bincount(e[v], minlength=513).astype(np.float64)   # complete obs ending at k
    cnt = np.bincount(e, minlength=513).astype(np.float64)
    n = cnt[::-1].cumsum()[::-1]                              # rows reaching length >= k
    h = np.where(n > 0, d / np.maximum(n, 1), 1.0)            # no data -> must end
    return np.clip(h, 1e-4, 1 - 1e-4)

h_train = hazard(train_idx)
h_val = hazard(val_idx)

def prior_logits(h, S):
    logit = np.log(h / (1 - h))
    return torch.from_numpy(logit[1:S + 1]).float()           # position i -> length i+1

def prior_bce(h, rows=None):
    rows = val_rows if rows is None else rows
    tot, n = 0.0, 0
    for r in rows:
        S = r["num_steps"]
        logits = prior_logits(h, S)
        labels = torch.zeros(S)
        if r["num_steps_true"] == S:
            labels[-1] = 1.0
        bce = F.binary_cross_entropy_with_logits(
            logits, labels, reduction="none",
            pos_weight=torch.tensor(POS_WEIGHT),
        )
        tot += bce.sum().item()
        n += S
    return tot / n

# trainer's decision metrics (first logit > 0) + argmax variant
def router_metrics(h, rule="first_positive"):
    S_max = max(S_lens)
    logits = torch.full((len(val_rows), S_max), float("-inf"))
    valid = torch.zeros(len(val_rows), S_max, dtype=torch.bool)
    for j, r in enumerate(val_rows):
        S = r["num_steps"]
        logits[j, :S] = prior_logits(h, S)
        valid[j, :S] = True
    masked = logits.masked_fill(~valid, float("-inf"))
    if rule == "first_positive":
        did_halt = (masked > 0).any(dim=1)
        pred_halt = torch.where(did_halt, (masked > 0).int().argmax(dim=1), -1)
    else:  # argmax: the prior's best guess, always halts
        did_halt = torch.ones(len(val_rows), dtype=torch.bool)
        pred_halt = masked.argmax(dim=1)

    ns = torch.tensor([r["num_steps"] for r in val_rows])
    ns_true = torch.tensor([r["num_steps_true"] for r in val_rows])
    halt_visible = (ns > 0) & (ns_true == ns)
    cut = ~halt_visible
    true_halt = ns - 1
    eval_halt = halt_visible & did_halt
    n_visible, n_halted = int(halt_visible.sum()), int(eval_halt.sum())

    out = {"halt_rate": n_halted / max(n_visible, 1)}
    if n_halted > 0:
        offset = (pred_halt - true_halt)[eval_halt].float()
        out["signed_off"] = offset.mean().item()
        out["abs_off"] = offset.abs().mean().item()
        out["median_off"] = offset.median().item()
        out["exact"] = (offset == 0).float().mean().item()
    n_cut = int(cut.sum())
    out["false_halt_rate"] = (cut & did_halt).sum().item() / max(n_cut, 1)
    halt_pos = torch.zeros_like(valid)
    rows_ = halt_visible.nonzero(as_tuple=True)[0]
    halt_pos[rows_, true_halt[rows_]] = True
    pred_pos = valid & (logits > 0)
    tp = int((halt_pos & pred_pos).sum()); fp = int((~halt_pos & pred_pos).sum())
    fn = int((halt_pos & ~pred_pos).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    out["f1"] = 2 * prec * rec / max(prec + rec, 1e-12)
    return out

print("=" * 72, flush=True)
print(f"REGRESSION  (model val reg @143k = 1.0769)", flush=True)
print(f"  constant mean   all positions : {const_loss(const_mean):.4f}")
print(f"  constant median all positions : {const_loss(const_median):.4f}")
print(f"  identity        steps 1..S-1  : {identity_loss(1):.4f}")
print(f"  constant mean   steps 1..S-1  : {const_loss(const_mean, 1):.4f}")
print(f"  identity        incl ctx@0    : {identity_loss(0):.4f}  (ctx is a different encoder space)", flush=True)
print("=" * 72, flush=True)
for name, h in (("length prior (train dist)", h_train), ("length prior (val dist, upper bound)", h_val)):
    print(f"  {name}", flush=True)
    print(f"    weighted bce : {prior_bce(h):.4f}")
    m1 = router_metrics(h, "first_positive")
    m2 = router_metrics(h, "argmax")
    print(f"    first-pos rule: halt_rate={m1['halt_rate']:.3f} abs_off={m1.get('abs_off', float('nan')):.1f} "
          f"f1={m1['f1']:.3f} false_halt={m1['false_halt_rate']:.3f}")
    print(f"    argmax rule  : halt_rate={m2['halt_rate']:.3f} abs_off={m2.get('abs_off', float('nan')):.1f} "
          f"f1={m2['f1']:.3f} false_halt={m2['false_halt_rate']:.3f}", flush=True)
print("DONE", flush=True)
