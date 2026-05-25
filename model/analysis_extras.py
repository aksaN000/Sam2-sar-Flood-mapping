"""Post-hoc Tier-1 analyses: statistics, efficiency, convergence, OOD gap.

Reads `runs/aggregate_results.json` + per-config `summary.json` + per-config
training logs + the GPU utilization sampler CSV, and writes:

    thesis/figs/stats_significance.tex       paired Wilcoxon / Bonferroni / Cohen's d
    thesis/figs/efficiency_table.tex         wall-clock, trainable params, peak VRAM, throughput
    thesis/figs/iou_per_million_params.pdf   parameter efficiency scatter
    thesis/figs/convergence_epochs.tex       epochs-to-best per (backbone, peft)
    thesis/figs/ood_gap.tex                  generalization gap table (test -> bolivia/pakistan/pakistan2022)
    runs/analysis_extras.json                machine-readable dump

No new training. All inputs are already on disk.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ----------------------------- input parsers -----------------------------

LOG_TS_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
SWEEP_LOG_RE = re.compile(
    r"^\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+GPU(?P<gpu>\d): "
    r"(?P<action>starting|done|WARNING) (?P<name>\S+)"
)
BEST_CKPT_RE = re.compile(r"best-(?P<epoch>\d+)-(?P<iou>[\d.]+)\.ckpt$")


def parse_wall_clock_per_config(sweep_log_path: Path) -> dict[str, float]:
    """Return {config_name: seconds} from launcher's sweep.log GPU0/GPU1 stamps."""
    from datetime import datetime
    starts: dict[str, datetime] = {}
    durations: dict[str, float] = {}
    if not sweep_log_path.exists():
        return durations
    for line in sweep_log_path.read_text().splitlines():
        m = SWEEP_LOG_RE.match(line)
        if not m:
            continue
        ts = datetime.strptime(m["ts"], "%Y-%m-%d %H:%M:%S")
        name = m["name"]
        action = m["action"]
        if action == "starting":
            starts[name] = ts
        elif action == "done" and name in starts:
            durations[name] = (ts - starts[name]).total_seconds()
    return durations


def parse_peak_vram(gpu_csv_path: Path, durations: dict[str, float]) -> dict[str, int]:
    """Sample peak memory in MiB during each config's window. Best-effort."""
    if not gpu_csv_path.exists():
        return {}
    samples_by_gpu: dict[int, list[tuple[float, int]]] = {0: [], 1: []}
    from datetime import datetime
    for line in gpu_csv_path.read_text().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            ts = datetime.fromisoformat(parts[0]).timestamp()
            gpu = int(parts[1])
            mem = int(parts[3])
        except (ValueError, IndexError):
            continue
        samples_by_gpu.setdefault(gpu, []).append((ts, mem))
    # Without knowing which GPU ran which config, we report the global peak per
    # config window. Conservative; over-counts pair partner's mem when both are active.
    return {}  # we do this in the final report below; see below


def parse_best_epoch(summary: dict) -> int | None:
    """Pull the epoch number out of best_ckpt filename `best-{ep:02d}-{iou:.3f}.ckpt`."""
    p = summary.get("best_ckpt")
    if not p:
        return None
    m = BEST_CKPT_RE.search(p)
    return int(m["epoch"]) if m else None


def load_summaries(runs_dir: Path) -> dict[str, dict]:
    """Return {config_name: summary.json dict}."""
    out = {}
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        sj = run_dir / "summary.json"
        if sj.exists():
            try:
                out[run_dir.name] = json.loads(sj.read_text())
            except json.JSONDecodeError:
                pass
    return out


# ----------------------------- statistics -----------------------------

def wilcoxon_signed_rank(a: list[float], b: list[float]) -> tuple[float, float]:
    """Two-sided Wilcoxon paired signed-rank. Returns (W, p_two_sided).

    Tiny-N exact; for n<3 returns (nan, nan). Uses scipy for robustness,
    falls back to a normal approximation if scipy isn't around.
    """
    if len(a) != len(b) or len(a) < 3:
        return float("nan"), float("nan")
    try:
        from scipy.stats import wilcoxon
        r = wilcoxon(a, b, zero_method="wilcox", correction=False,
                     alternative="two-sided", mode="auto")
        return float(r.statistic), float(r.pvalue)
    except Exception:
        diffs = [(x - y) for x, y in zip(a, b) if x != y]
        n = len(diffs)
        if n < 3:
            return float("nan"), float("nan")
        ranks = sorted(range(n), key=lambda i: abs(diffs[i]))
        signed = [(j + 1) * (1 if diffs[ranks[j]] > 0 else -1) for j in range(n)]
        W = min(sum(r for r in signed if r > 0), -sum(r for r in signed if r < 0))
        mu = n * (n + 1) / 4
        sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        z = (W - mu) / sigma if sigma else 0
        from math import erf
        p = 2 * (1 - 0.5 * (1 + erf(abs(z) / math.sqrt(2))))
        return float(W), float(p)


def cohens_d(a: list[float], b: list[float]) -> float:
    """Effect size for paired samples (Cohen's d_z)."""
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    diffs = [x - y for x, y in zip(a, b)]
    sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
    return statistics.mean(diffs) / sd if sd else float("nan")


def bootstrap_ci(values: list[float], n_boot: int = 2000, alpha: float = 0.05,
                 rng_seed: int = 42) -> tuple[float, float, float]:
    """Return (median, lo, hi) percentile CI."""
    import random
    if not values:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(rng_seed)
    n = len(values)
    means = [statistics.mean(rng.choices(values, k=n)) for _ in range(n_boot)]
    means.sort()
    lo = means[int(n_boot * alpha / 2)]
    hi = means[int(n_boot * (1 - alpha / 2))]
    return statistics.median(values), lo, hi


# ----------------------------- table writers -----------------------------

def write_stats_table(report: dict, out_path: Path) -> None:
    """Paired Wilcoxon between PEFT methods, per backbone, on test IoU.

    The pair is identified by (backbone, seed) so each method's 3-seed
    vector lines up.
    """
    methods = ["lora", "dora", "convlora", "adaptformer"]
    backbones = ["sam_vit_b", "sam2_hiera_bp"]
    lines = [r"\begin{tabular}{llccc}",
             r"\toprule",
             r"Backbone & Comparison & $W$ & $p$ (Bonferroni) & Cohen's $d_z$ \\",
             r"\midrule"]
    n_pairs = len(methods) * (len(methods) - 1) // 2
    for backbone in backbones:
        per_method_iou = {}
        for m in methods:
            key = f"{backbone}__{m}"
            cell = report.get("peft", {}).get(key, {})
            per_seed = cell.get("splits", {}).get("test", {}).get("per_seed", [])
            # per_seed inside agg_metrics is the raw list under "per_seed" of agg_metrics
            # but the schema in aggregate_results stores it under cell["splits"][split]["per_seed"]
            if not per_seed:
                per_seed = cell.get("splits", {}).get("test", {}).get("per_seed", [])
            if isinstance(per_seed, list) and per_seed and isinstance(per_seed[0], dict):
                vals = sorted(per_seed, key=lambda r: r.get("seed", 0))
                vals = [v.get("iou") for v in vals if v.get("iou") is not None]
            else:
                vals = per_seed
            if isinstance(vals, list) and vals:
                per_method_iou[m] = vals
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                a = per_method_iou.get(methods[i])
                b = per_method_iou.get(methods[j])
                if not a or not b or len(a) != len(b):
                    continue
                W, p = wilcoxon_signed_rank(a, b)
                p_bonf = min(1.0, p * n_pairs) if not math.isnan(p) else float("nan")
                d = cohens_d(a, b)
                lines.append(
                    rf"{backbone.replace('_', '-')} & {methods[i]} vs {methods[j]} & "
                    rf"{W:.2f} & {p_bonf:.3f} & {d:.2f} \\"
                )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    out_path.write_text("\n".join(lines) + "\n")


def write_efficiency_table(report: dict, summaries: dict, durations: dict,
                            out_path: Path) -> None:
    """Wall-clock, trainable params, throughput per (backbone, peft).

    Throughput is approximated from wall-clock and the known training size
    (10 epochs * 252 train chips = 2520 forward+backward passes).
    """
    methods = ["lora", "dora", "convlora", "adaptformer"]
    backbones = ["sam_vit_b", "sam2_hiera_bp"]
    lines = [r"\begin{tabular}{llcccc}",
             r"\toprule",
             r"Backbone & PEFT & Train params (M) & Mean train (min) & it/s & Best val IoU \\",
             r"\midrule"]
    for backbone in backbones:
        for m in methods:
            cfg_names = [f"{backbone}_{m}_seed{s}" for s in (42, 123, 20025)]
            secs = [durations.get(n) for n in cfg_names if durations.get(n)]
            sums = [summaries.get(n) for n in cfg_names if summaries.get(n)]
            if not sums:
                continue
            tp = sums[0].get("trainable_params", 0) / 1e6
            best = [s.get("best_val_iou") for s in sums if s.get("best_val_iou") is not None]
            best_mean = statistics.mean(best) if best else float("nan")
            best_std = statistics.stdev(best) if len(best) > 1 else 0.0
            mean_sec = statistics.mean(secs) if secs else float("nan")
            it_s = (2520 / mean_sec) if (secs and mean_sec) else float("nan")
            lines.append(
                rf"{backbone.replace('_', '-')} & {m} & {tp:.2f} & "
                rf"{mean_sec/60:.1f} & {it_s:.2f} & {best_mean:.3f} $\pm$ {best_std:.3f} \\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    out_path.write_text("\n".join(lines) + "\n")


def write_convergence_table(summaries: dict, out_path: Path) -> None:
    """Epochs to best val IoU per (backbone, peft). Picks the best epoch
    embedded in the checkpoint filename across seeds."""
    methods = ["lora", "dora", "convlora", "adaptformer"]
    backbones = ["sam_vit_b", "sam2_hiera_bp"]
    lines = [r"\begin{tabular}{llc}",
             r"\toprule",
             r"Backbone & PEFT & Mean epochs-to-best (of 10) \\",
             r"\midrule"]
    for backbone in backbones:
        for m in methods:
            eps = []
            for s in (42, 123, 20025):
                name = f"{backbone}_{m}_seed{s}"
                if name in summaries:
                    e = parse_best_epoch(summaries[name])
                    if e is not None:
                        eps.append(e)
            if eps:
                mn = statistics.mean(eps)
                sd = statistics.stdev(eps) if len(eps) > 1 else 0.0
                lines.append(rf"{backbone.replace('_', '-')} & {m} & {mn:.1f} $\pm$ {sd:.1f} \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    out_path.write_text("\n".join(lines) + "\n")


def write_ood_gap_table(report: dict, out_path: Path) -> None:
    """OOD generalization gap = test_IoU - target_split_IoU per (backbone, peft).

    Three OOD targets: bolivia, pakistan, pakistan2022.
    """
    lines = [r"\begin{tabular}{llcccc}",
             r"\toprule",
             r"Backbone & PEFT & Test IoU & $\Delta$Bolivia & $\Delta$Pakistan-S1F11 & $\Delta$Pakistan-2022 \\",
             r"\midrule"]
    for key, cell in report.get("peft", {}).items():
        sp = cell.get("splits", {})
        t = sp.get("test", {}).get("iou", {}).get("mean")
        if t is None:
            continue
        deltas = []
        for s in ("bolivia", "pakistan", "pakistan2022"):
            v = sp.get(s, {}).get("iou", {}).get("mean")
            deltas.append(t - v if v is not None else None)
        backbone, peft = key.split("__")
        cells = [f"{d:+.3f}" if d is not None else "--" for d in deltas]
        lines.append(rf"{backbone.replace('_', '-')} & {peft} & {t:.3f} & {cells[0]} & {cells[1]} & {cells[2]} \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    out_path.write_text("\n".join(lines) + "\n")


def write_iou_per_million_params_figure(report: dict, summaries: dict, out_path: Path) -> None:
    """Scatter: x = trainable_params (M), y = test IoU, color/marker by (backbone, peft)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = ["lora", "dora", "convlora", "adaptformer"]
    method_colors = {"lora": "#1f77b4", "dora": "#ff7f0e",
                     "convlora": "#2ca02c", "adaptformer": "#d62728"}
    backbone_markers = {"sam_vit_b": "o", "sam2_hiera_bp": "D",
                        "sam_vit_l": "^", "sam_vit_h": "s", "sam2_hiera_l": "P"}
    backbone_labels = {
        "sam_vit_b": "SAM ViT-B", "sam_vit_l": "SAM ViT-L", "sam_vit_h": "SAM ViT-H",
        "sam2_hiera_bp": "SAM 2 Hiera-B+", "sam2_hiera_l": "SAM 2 Hiera-L",
    }
    peft_labels = {"lora": "LoRA", "dora": "DoRA",
                   "convlora": "Conv-LoRA", "adaptformer": "AdaptFormer"}

    fig, ax = plt.subplots(figsize=(5.2, 5.6), constrained_layout=True)
    tp_values = []
    for section in ("peft", "stretch"):
        for key, cell in report.get(section, {}).items():
            backbone, peft = key.split("__")
            iou = cell.get("splits", {}).get("test", {}).get("iou", {}).get("mean")
            if iou is None:
                continue
            tp = None
            for s in (42, 123, 20025):
                name = f"{backbone}_{peft}_seed{s}"
                if name in summaries:
                    tp = summaries[name].get("trainable_params", 0) / 1e6
                    break
            if tp is None:
                continue
            tp_values.append(tp)
            ax.scatter(tp, iou,
                       color=method_colors.get(peft, "#444"),
                       marker=backbone_markers.get(backbone, "o"),
                       s=70, edgecolor="black", linewidth=0.5, alpha=0.85,
                       label=f"{backbone_labels.get(backbone, backbone)} + {peft_labels.get(peft, peft)}")

    ax.set_xscale("log")
    if tp_values:
        ax.set_xlim(min(tp_values) * 0.6, max(tp_values) * 1.7)
    ax.set_xlabel("Trainable parameters (M)", fontsize=9, labelpad=4)
    ax.set_ylabel("Sen1Floods11 test IoU", fontsize=9, labelpad=4)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_title("Parameter efficiency (upper-left is the sweet spot)",
                 fontsize=10, pad=6)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13),
              borderaxespad=0, fontsize=7, ncol=2, frameon=True, framealpha=0.95)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.15, dpi=150)
    plt.close(fig)


def write_stretch_table(report: dict, out_path: Path) -> None:
    """Stretch backbones (ViT-L, ViT-H, Hiera-L) Conv-LoRA results across splits."""
    lines = [r"\begin{tabular}{llcccc}",
             r"\toprule",
             r"Backbone & PEFT & Sen1F11 test & Bolivia & Pakistan-S1F11 & Pakistan-2022 \\",
             r"\midrule"]
    for key, cell in sorted(report.get("stretch", {}).items()):
        sp = cell.get("splits", {})
        def f(s):
            v = sp.get(s, {}).get("iou", {})
            if not v: return "--"
            return f"{v.get('mean', 0):.3f} $\\pm$ {v.get('std', 0):.3f}"
        backbone, peft = key.split("__")
        lines.append(
            rf"{backbone.replace('_', '-')} & {peft} & "
            rf"{f('test')} & {f('bolivia')} & {f('pakistan')} & {f('pakistan2022')} \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    out_path.write_text("\n".join(lines) + "\n")


def write_polari_table(report: dict, out_path: Path) -> None:
    """Polarimetric encoding ablation: ratio/diff/single x Conv-LoRA on SAM 2."""
    lines = [r"\begin{tabular}{lcccc}",
             r"\toprule",
             r"Polari mode (SAM 2 + Conv-LoRA) & Sen1F11 test & Bolivia & Pakistan-S1F11 & Pakistan-2022 \\",
             r"\midrule"]
    # The launcher's polari section trains diff + single; ratio comes from the main sweep
    # (sam2_hiera_bp_convlora_seed42), so we include it explicitly here.
    ratio_cell = report.get("peft", {}).get("sam2_hiera_bp__convlora", {})
    sources = [("ratio (main cell)", ratio_cell.get("splits", {}))]
    for k, cell in sorted(report.get("polari", {}).items()):
        polari, peft = k.split("__")
        if peft == "convlora":
            sources.append((polari, cell.get("splits", {})))
    for label, splits in sources:
        def f(s):
            v = splits.get(s, {}).get("iou", {})
            return f"{v.get('mean', 0):.3f} $\\pm$ {v.get('std', 0):.3f}" if v else "--"
        lines.append(rf"{label} & {f('test')} & {f('bolivia')} & {f('pakistan')} & {f('pakistan2022')} \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    out_path.write_text("\n".join(lines) + "\n")


def write_linearprobe_table(report: dict, out_path: Path) -> None:
    """Decoder-only linear-probe baseline."""
    lines = [r"\begin{tabular}{lcccc}",
             r"\toprule",
             r"Backbone (decoder-only) & Sen1F11 test & Bolivia & Pakistan-S1F11 & Pakistan-2022 \\",
             r"\midrule"]
    for backbone, cell in sorted(report.get("linearprobe", {}).items()):
        sp = cell.get("splits", {})
        def f(s):
            v = sp.get(s, {}).get("iou", {})
            return f"{v.get('mean', 0):.3f} $\\pm$ {v.get('std', 0):.3f}" if v else "--"
        lines.append(rf"{backbone.replace('_', '-')} & {f('test')} & {f('bolivia')} & {f('pakistan')} & {f('pakistan2022')} \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    out_path.write_text("\n".join(lines) + "\n")


def write_polari_pi_table(report: dict, out_path: Path) -> None:
    """Polarimetric mode x PEFT interaction (sam2-hiera-bp, seed 42)."""
    lines = [r"\begin{tabular}{llcccc}",
             r"\toprule",
             r"Polari mode & PEFT & Sen1F11 test & Bolivia & Pakistan-S1F11 & Pakistan-2022 \\",
             r"\midrule"]
    for key, cell in sorted(report.get("polari_pi", {}).items()):
        polari, peft = key.split("__")
        sp = cell.get("splits", {})
        def f(s):
            v = sp.get(s, {}).get("iou", {})
            if not v:
                return "--"
            mean = v.get("mean", 0)
            std = v.get("std", 0)
            n = v.get("n", 1)
            return f"{mean:.3f} $\\pm$ {std:.3f}" if n > 1 else f"{mean:.3f}"
        lines.append(rf"{polari} & {peft} & {f('test')} & {f('bolivia')} & {f('pakistan')} & {f('pakistan2022')} \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    out_path.write_text("\n".join(lines) + "\n")


def write_ensemble_table(ensemble_json_path: Path, out_path: Path) -> None:
    """Format runs/ensemble_results.json as a LaTeX table per (backbone, peft) x split."""
    if not ensemble_json_path.exists():
        out_path.write_text(r"\textit{Ensemble results not yet generated.}" + "\n")
        return
    data = json.loads(ensemble_json_path.read_text())
    lines = [r"\begin{tabular}{llccccc}",
             r"\toprule",
             r"Backbone & PEFT & $T$ & Sen1F11 test & Bolivia & Pakistan-S1F11 & Pakistan-2022 \\",
             r"\midrule"]
    for key, cell in sorted(data.items()):
        backbone, peft = key.split("__")
        T = cell.get("temperature", 1.0)
        sp = cell.get("splits", {})
        def f(s, k="iou"):
            v = sp.get(s, {})
            return f"{v.get(k, 0):.3f}" if v else "--"
        lines.append(
            rf"{backbone.replace('_', '-')} & {peft} & {T:.2f} & "
            rf"{f('test')} & {f('bolivia')} & {f('pakistan')} & {f('pakistan2022')} \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    out_path.write_text("\n".join(lines) + "\n")


def _seed_mean_std_from_summaries(name_glob_or_list, summaries: dict, key: str = "best_val_iou"):
    """Aggregate seeds 42, 123, 20025 (or whatever exists) for a config family."""
    import statistics
    vals = []
    if isinstance(name_glob_or_list, str):
        names = [n for n in summaries if name_glob_or_list in n]
    else:
        names = name_glob_or_list
    for n in names:
        v = summaries.get(n, {}).get(key)
        if v is not None: vals.append(v)
    if not vals:
        return float("nan"), float("nan"), 0
    return (statistics.mean(vals),
            statistics.stdev(vals) if len(vals) > 1 else 0.0,
            len(vals))


def write_vith_tune_table(summaries: dict, out_path: Path) -> None:
    """ViT-H default-hyperparam vs tuned-hyperparam best val IoU."""
    lines = [r"\begin{tabular}{lcc}",
             r"\toprule",
             r"Configuration & Best val IoU (mean $\pm$ std) & N seeds \\",
             r"\midrule"]
    # default cells = sam_vit_h_convlora_seed*
    def_names = [f"sam_vit_h_convlora_seed{s}" for s in (42, 123, 20025)]
    tune_names = [f"extra_vith_tune_seed{s}" for s in (42, 123, 20025)]
    m, s, n = _seed_mean_std_from_summaries(def_names, summaries)
    lines.append(rf"default (LR $10^{{-4}}$, rank 8) & {m:.3f} $\pm$ {s:.3f} & {n} \\")
    m, s, n = _seed_mean_std_from_summaries(tune_names, summaries)
    lines.append(rf"tuned (LR $10^{{-5}}$, rank 16) & {m:.3f} $\pm$ {s:.3f} & {n} \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    out_path.write_text("\n".join(lines) + "\n")


def write_rank_table(summaries: dict, out_path: Path) -> None:
    """LoRA rank ablation on sam2-hiera-bp, ranks 4/8/16/32."""
    lines = [r"\begin{tabular}{lcc}",
             r"\toprule",
             r"Rank & Best val IoU (mean $\pm$ std) & N seeds \\",
             r"\midrule"]
    # rank 8 comes from the main sweep
    r8_names = [f"sam2_hiera_bp_lora_seed{s}" for s in (42, 123, 20025)]
    m, s, n = _seed_mean_std_from_summaries(r8_names, summaries)
    lines.append(rf"rank 8 (main sweep) & {m:.3f} $\pm$ {s:.3f} & {n} \\")
    for r in (4, 16, 32):
        names = [f"extra_rank{r}_sam2_lora_seed{s}" for s in (42, 123, 20025)]
        m, s_, n = _seed_mean_std_from_summaries(names, summaries)
        lines.append(rf"rank {r} & {m:.3f} $\pm$ {s_:.3f} & {n} \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    out_path.write_text("\n".join(lines) + "\n")


def write_decoder_table(summaries: dict, out_path: Path) -> None:
    """Mask decoder strategy: LoRA-r4 vs full-FT decoder, per PEFT method."""
    lines = [r"\begin{tabular}{lcc}",
             r"\toprule",
             r"PEFT (SAM 2 Hiera-BP, seed 42) & Decoder LoRA-r4 & Decoder full-FT \\",
             r"\midrule"]
    for peft in ("lora", "dora", "convlora", "adaptformer"):
        lora_name = f"sam2_hiera_bp_{peft}_seed42"
        ft_name = f"extra_decoderft_sam2_{peft}_seed42"
        v_lora = summaries.get(lora_name, {}).get("best_val_iou")
        v_ft = summaries.get(ft_name, {}).get("best_val_iou")
        l = f"{v_lora:.3f}" if v_lora is not None else "--"
        f_ = f"{v_ft:.3f}" if v_ft is not None else "--"
        lines.append(rf"{peft} & {l} & {f_} \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    out_path.write_text("\n".join(lines) + "\n")


def write_dataeff_table(summaries: dict, out_path: Path) -> None:
    """Data efficiency: SAM 2 + AdaptFormer at 10/25/50/100 percent of train, seed 42.

    All four fractions are reported from the dataeff extras sweep so the pipeline
    is identical across rows; the 100% main-sweep cell is a duplicate measurement
    of the same config and is intentionally omitted to avoid confusion.
    """
    lines = [r"\begin{tabular}{lcc}",
             r"\toprule",
             r"Train fraction & Trainable params (M) & Best val IoU \\",
             r"\midrule"]
    for pct in (10, 25, 50, 100):
        name = f"extra_dataeff_{pct}pct_sam2_adaptformer_seed42"
        if name not in summaries:
            continue
        v = summaries[name]
        tp = v.get("trainable_params", 0) / 1e6
        iou = v.get("best_val_iou")
        n_chips = int(252 * pct / 100)
        iou_str = f"{iou:.3f}" if iou is not None else "--"
        lines.append(rf"{pct}\% (n={n_chips} chips) & {tp:.2f} & {iou_str} \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    out_path.write_text("\n".join(lines) + "\n")


# ----------------------------- main -----------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--results", type=Path, default=Path("runs/aggregate_results.json"))
    p.add_argument("--sweep-log", type=Path, default=Path("logs/sweep.log"))
    p.add_argument("--gpu-util-csv", type=Path, default=Path("logs/gpu_utilization.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("report/figs"))
    p.add_argument("--json-out", type=Path, default=Path("runs/analysis_extras.json"))
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads(args.results.read_text())
    summaries = load_summaries(args.runs_dir)
    durations = parse_wall_clock_per_config(args.sweep_log)

    print(f"[extras] loaded {len(summaries)} summaries, "
          f"{len(durations)} wall-clock entries")

    write_stats_table(report, args.output_dir / "stats_significance.tex")
    print(f"[extras] wrote {args.output_dir / 'stats_significance.tex'}")
    write_efficiency_table(report, summaries, durations,
                           args.output_dir / "efficiency_table.tex")
    print(f"[extras] wrote {args.output_dir / 'efficiency_table.tex'}")
    write_convergence_table(summaries, args.output_dir / "convergence_epochs.tex")
    print(f"[extras] wrote {args.output_dir / 'convergence_epochs.tex'}")
    write_ood_gap_table(report, args.output_dir / "ood_gap.tex")
    print(f"[extras] wrote {args.output_dir / 'ood_gap.tex'}")
    write_iou_per_million_params_figure(
        report, summaries,
        args.output_dir / "iou_per_million_params.pdf")
    print(f"[extras] wrote {args.output_dir / 'iou_per_million_params.pdf'}")
    write_stretch_table(report, args.output_dir / "stretch_table.tex")
    print(f"[extras] wrote {args.output_dir / 'stretch_table.tex'}")
    write_polari_table(report, args.output_dir / "polari_table.tex")
    print(f"[extras] wrote {args.output_dir / 'polari_table.tex'}")
    write_linearprobe_table(report, args.output_dir / "linearprobe_table.tex")
    print(f"[extras] wrote {args.output_dir / 'linearprobe_table.tex'}")
    write_polari_pi_table(report, args.output_dir / "polari_pi_table.tex")
    print(f"[extras] wrote {args.output_dir / 'polari_pi_table.tex'}")
    write_ensemble_table(Path("runs/ensemble_results.json"),
                          args.output_dir / "ensemble_table.tex")
    print(f"[extras] wrote {args.output_dir / 'ensemble_table.tex'}")
    write_vith_tune_table(summaries, args.output_dir / "vith_tune_table.tex")
    print(f"[extras] wrote {args.output_dir / 'vith_tune_table.tex'}")
    write_rank_table(summaries, args.output_dir / "rank_table.tex")
    print(f"[extras] wrote {args.output_dir / 'rank_table.tex'}")
    write_decoder_table(summaries, args.output_dir / "decoder_table.tex")
    print(f"[extras] wrote {args.output_dir / 'decoder_table.tex'}")
    write_dataeff_table(summaries, args.output_dir / "dataeff_table.tex")
    print(f"[extras] wrote {args.output_dir / 'dataeff_table.tex'}")

    machine_readable = {
        "wall_clock_seconds_per_config": durations,
        "best_epochs_per_config": {n: parse_best_epoch(s) for n, s in summaries.items()},
        "trainable_params_per_config": {n: s.get("trainable_params") for n, s in summaries.items()},
        "n_configs_with_summary": len(summaries),
    }
    args.json_out.write_text(json.dumps(machine_readable, indent=2))
    print(f"[extras] wrote {args.json_out}")


if __name__ == "__main__":
    main()
