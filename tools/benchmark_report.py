"""Shared report writer for benchmark rows.

Each benchmark row writes into its own folder under ``logs/benchmark``:
    meta.json        - machine-readable metrics + run config
    predictions.jsonl - one line per question (gold, predicted, raw, timing)
    report.md        - human-readable summary (accuracy, timing, context)

All three are produced by ``write_report``; scripts only assemble the
numbers and the markdown body.
"""
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def new_run_dir(benchmark: str, prefix: str, run_name: str = None) -> Path:
    """Return ``logs/benchmark/<benchmark>/<prefix>_<timestamp>`` (created)."""
    name = (run_name or prefix) + "_" + time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path("logs") / "benchmark" / benchmark / name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_report(out_dir: Path, meta: Dict[str, Any],
                 predictions: Optional[List[Dict[str, Any]]],
                 report_md: str) -> Path:
    """Write meta.json, predictions.jsonl and report.md into *out_dir*.

    Pass ``predictions=None`` when the calling script already wrote
    predictions.jsonl incrementally (append mode during the run) - the
    file is then left untouched.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    if predictions is not None:
        with open(out_dir / "predictions.jsonl", "w") as f:
            for p in predictions:
                f.write(json.dumps(p) + "\n")
    (out_dir / "report.md").write_text(report_md)
    return out_dir


def timing_summary(latencies: List[float], tokens: List[int]) -> Dict[str, Any]:
    """Mean/median/p95 helpers for latency and decode-token lists."""
    import statistics

    lat = sorted(latencies)
    tok = sorted(tokens)
    n = max(len(lat), 1)

    def p95(xs):
        return xs[min(int(0.95 * len(xs)), len(xs) - 1)] if xs else 0.0

    return {
        "latency_mean_s": round(statistics.mean(lat), 3),
        "latency_median_s": round(statistics.median(lat), 3),
        "latency_p95_s": round(p95(lat), 3),
        "latency_total_s": round(sum(lat), 1),
        "decode_tokens_mean": round(statistics.mean(tok), 1),
        "decode_tokens_median": int(statistics.median(tok)),
        "decode_tokens_p95": int(p95(tok)),
        "n_timed": n,
    }


def format_timing_md(ts: Dict[str, Any]) -> str:
    return (
        "| metric | value |\n|---|---|\n"
        f"| latency mean / median / p95 | {ts['latency_mean_s']}s / "
        f"{ts['latency_median_s']}s / {ts['latency_p95_s']}s |\n"
        f"| total wall time | {ts['latency_total_s']}s |\n"
        f"| decode tokens mean / median / p95 | {ts['decode_tokens_mean']} / "
        f"{ts['decode_tokens_median']} / {ts['decode_tokens_p95']} |\n"
    )
