"""Generate figures and LaTeX-ready tables from runs/aggregate_results.json.

Produces:

    figs/results_table.tex            Headline IoU/F1/precision/recall table
                                      (replaces tab:expected_main placeholders).
    figs/iou_vs_bolivia.pdf           Scatter of Sen1Floods11 test IoU against
                                      Bolivia OOD IoU, one marker per
                                      (backbone, PEFT) combination.
    figs/ood_gap.pdf                  Bar plot of the OOD generalization gap
                                      (test_iou - bolivia_iou) per method.
    figs/peft_summary.pdf             Grouped bar chart of test/bolivia IoU
                                      per (backbone, PEFT).
    figs/reliability_diagram.pdf      Calibration plot (if confidence eval ran).

CLI
---
    python -m model.make_figures \\
        --results runs/aggregate_results.json \\
        --output thesis/figs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend; no display required
import matplotlib.pyplot as plt
import numpy as np


# Pretty labels for column / row headers.
BACKBONE_LABELS = {
    "sam_vit_b":     "SAM ViT-B",
    "sam_vit_l":     "SAM ViT-L",
    "sam_vit_h":     "SAM ViT-H",
    "sam2_hiera_t":  r"SAM 2 Hiera-T",
    "sam2_hiera_s":  r"SAM 2 Hiera-S",
    "sam2_hiera_bp": r"SAM 2 Hiera-B+",
    "sam2_hiera_l":  r"SAM 2 Hiera-L",
}
PEFT_LABELS = {
    "lora":        "LoRA",
    "dora":        "DoRA",
    "convlora":    "Conv-LoRA",
    "adaptformer": "AdaptFormer",
}

# Per-method colors used consistently across all figures so the reader can
# track a method visually from one plot to the next.
PEFT_COLORS = {
    "lora":        "#1f77b4",   # blue
    "dora":        "#ff7f0e",   # orange
    "convlora":    "#2ca02c",   # green
    "adaptformer": "#d62728",   # red
}
BASELINE_COLOR = "#7f7f7f"      # gray for U-Net
ZEROSHOT_COLOR = "#bcbcbc"      # light gray for zero-shot

# Per-backbone marker shapes; reader can tell ViT-B from Hiera-B+ at a glance.
BACKBONE_MARKERS = {
    "sam_vit_b":     "o",       # circle
    "sam_vit_l":     "^",       # triangle up
    "sam_vit_h":     "s",       # square (the largest SAM)
    "sam2_hiera_t":  "v",       # triangle down
    "sam2_hiera_s":  "<",       # triangle left
    "sam2_hiera_bp": "D",       # diamond (our headline)
    "sam2_hiera_l":  ">",       # triangle right (the largest SAM 2)
}


def _set_paper_style() -> None:
    """Set matplotlib rcParams for paper-quality figures.

    Called once at the top of main(). Effects: larger fonts (readable at
    journal-column width), higher DPI, vector text in saved PDFs.
    """
    plt.rcParams.update({
        "font.size":         11,
        "axes.titlesize":    12,
        "axes.labelsize":    11,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "legend.fontsize":    9,
        "figure.dpi":       100,
        "savefig.dpi":      200,
        "savefig.bbox":   "tight",
        "axes.grid":         True,
        "grid.alpha":         0.3,
        "grid.linestyle":  "--",
        "axes.axisbelow":   True,   # grid behind data
        "axes.spines.top":  False,
        "axes.spines.right":False,
        "pdf.fonttype":     42,     # embed fonts as TrueType so they're searchable
    })


def parse_peft_key(key: str) -> tuple[str, str]:
    """Split aggregate-results key `backbone__peft` into the two parts."""
    backbone, peft = key.split("__")
    return backbone, peft


def render_results_table(report: dict) -> str:
    """LaTeX `tabular` body for the principal results table.

    Columns: Backbone | Method | Sen1F11 test | Bolivia held-out | Pakistan-Sen1F11 | Pakistan-2022
    """
    lines: list[str] = []
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Backbone} & \textbf{Method} & \textbf{Sen1F11 test} & \textbf{Bolivia held-out} & \textbf{Pakistan-Sen1F11} & \textbf{Pakistan-2022} \\")
    lines.append(r"                  &                & IoU $\pm$ std         & IoU $\pm$ std            & IoU $\pm$ std            & IoU $\pm$ std         \\")
    lines.append(r"\midrule")

    def cells_for(splits: dict) -> tuple[str, str, str, str]:
        t = splits.get("test", {}).get("iou", {})
        b = splits.get("bolivia", {}).get("iou", {})
        pk = splits.get("pakistan", {}).get("iou", {})
        pk22 = splits.get("pakistan2022", {}).get("iou", {})
        return fmt_pm(t), fmt_pm(b), fmt_pm(pk), fmt_pm(pk22)

    if report.get("unet") and "splits" in report["unet"]:
        c1, c2, c3, c4 = cells_for(report["unet"]["splits"])
        lines.append(rf"U-Net (baseline) & full FT & {c1} & {c2} & {c3} & {c4} \\")

    if report.get("zeroshot"):
        for backbone, zs in sorted(report["zeroshot"].items()):
            label = BACKBONE_LABELS.get(backbone.replace("-", "_").replace("sam_vit", "sam_vit_b"),
                                        backbone)
            splits = zs.get("splits", {})
            t = splits.get("test", {})
            b = splits.get("bolivia", {})
            pk = splits.get("pakistan", {})
            pk22 = splits.get("pakistan2022", {})
            t_iou = f"{t['iou']:.3f}" if t else "--"
            b_iou = f"{b['iou']:.3f}" if b else "--"
            pk_iou = f"{pk['iou']:.3f}" if pk else "--"
            pk22_iou = f"{pk22['iou']:.3f}" if pk22 else "--"
            lines.append(rf"{label} & zero-shot & {t_iou} & {b_iou} & {pk_iou} & {pk22_iou} \\")

    # PEFT cells, ordered backbone-then-method.
    order = [(b, p) for b in ("sam_vit_b", "sam2_hiera_bp")
             for p in ("lora", "dora", "convlora", "adaptformer")]
    for backbone, peft in order:
        key = f"{backbone}__{peft}"
        if key not in report.get("peft", {}):
            continue
        c1, c2, c3, c4 = cells_for(report["peft"][key]["splits"])
        lines.append(rf"{BACKBONE_LABELS[backbone]} & {PEFT_LABELS[peft]} "
                     rf"& {c1} & {c2} & {c3} & {c4} \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def fmt_pm(stat: dict) -> str:
    if not stat:
        return "--"
    return f"${stat['mean']:.3f} \\pm {stat['std']:.3f}$"


def iou_scatter(report: dict, out_path: Path) -> None:
    """Scatter: x = test IoU, y = bolivia IoU. Diagonal = perfect generalization.

    Per-method color from PEFT_COLORS so the reader can track methods across
    figures. Per-backbone marker from BACKBONE_MARKERS so backbones are
    visually distinct without color collision.
    """
    fig, ax = plt.subplots(figsize=(5.2, 5.8), constrained_layout=True)
    # Small deterministic visual offset per PEFT to separate points when
    # methods cluster within a few hundredths of IoU. Marker positions shift
    # by this amount only; xerr / yerr still convey the real data variance.
    OFFSET_RADIUS = 0.008
    PEFT_OFFSETS = {
        "lora":        (+OFFSET_RADIUS, 0.0),
        "dora":        (0.0, +OFFSET_RADIUS),
        "convlora":    (-OFFSET_RADIUS, 0.0),
        "adaptformer": (0.0, -OFFSET_RADIUS),
    }
    for key, cell in report.get("peft", {}).items():
        backbone, peft = parse_peft_key(key)
        splits = cell["splits"]
        if "test" not in splits or "bolivia" not in splits:
            continue
        t = splits["test"]["iou"]["mean"]
        b = splits["bolivia"]["iou"]["mean"]
        t_std = splits["test"]["iou"]["std"]
        b_std = splits["bolivia"]["iou"]["std"]
        dx, dy = PEFT_OFFSETS.get(peft, (0.0, 0.0))
        color = PEFT_COLORS.get(peft, "#444")
        marker = BACKBONE_MARKERS.get(backbone, "o")
        ax.errorbar(t + dx, b + dy, xerr=t_std, yerr=b_std,
                    marker=marker, markersize=7, color=color,
                    markeredgecolor="black", markeredgewidth=0.5,
                    capsize=2, linewidth=1.0, alpha=0.85,
                    label=f"{BACKBONE_LABELS.get(backbone, backbone)} + {PEFT_LABELS.get(peft, peft)}")

    # U-Net baseline
    if report.get("unet"):
        u = report["unet"]["splits"]
        if "test" in u and "bolivia" in u:
            ax.plot(u["test"]["iou"]["mean"], u["bolivia"]["iou"]["mean"],
                    marker="*", markersize=14, color=BASELINE_COLOR,
                    markeredgecolor="black", markeredgewidth=0.8,
                    linestyle="None", label="U-Net (ResNet-34)")

    # Zero-shot floor
    if report.get("zeroshot"):
        for backbone, zs in report["zeroshot"].items():
            sp = zs.get("splits", {})
            if "test" in sp and "bolivia" in sp:
                label = BACKBONE_LABELS.get(backbone.replace("-", "_"), backbone)
                ax.plot(sp["test"]["iou"], sp["bolivia"]["iou"],
                        marker="x", markersize=11, color=ZEROSHOT_COLOR,
                        markeredgewidth=2, linestyle="None",
                        label=f"{label} zero-shot")

    lo = 0.0
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1], 0.5)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="identity")
    ax.set_xlabel("Sen1Floods11 test IoU", fontsize=9)
    ax.set_ylabel("Bolivia held-out IoU", fontsize=9)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.tick_params(axis="both", labelsize=8)
    ax.set_title("In-dist. vs. OOD IoU (markers nudged $\\pm 0.008$)",
                 fontsize=10, pad=6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13),
              borderaxespad=0, fontsize=7, ncol=2, frameon=True, framealpha=0.95)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.15, dpi=150)
    plt.close(fig)


def ood_gap_bars(report: dict, out_path: Path) -> None:
    """Bar plot: in-dist IoU minus OOD IoU per method. Smaller is better."""
    labels, gaps, colors = [], [], []
    for key, cell in report.get("peft", {}).items():
        backbone, peft = parse_peft_key(key)
        sp = cell["splits"]
        if "test" not in sp or "bolivia" not in sp:
            continue
        gap = sp["test"]["iou"]["mean"] - sp["bolivia"]["iou"]["mean"]
        labels.append(f"{BACKBONE_LABELS.get(backbone, backbone)}\n{PEFT_LABELS.get(peft, peft)}")
        gaps.append(gap)
        colors.append(PEFT_COLORS.get(peft, "#444"))
    fig, ax = plt.subplots(figsize=(5.0, 4.2), constrained_layout=True)
    bars = ax.bar(range(len(labels)), gaps, color=colors,
                  edgecolor="black", linewidth=0.5)
    short = [l.replace("SAM 2 Hiera-B+\n", "H+ ").replace("SAM ViT-B\n", "ViT-B ")
             for l in labels]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(short, rotation=35, ha="right", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylabel("Test IoU $-$ Bolivia IoU", fontsize=9, labelpad=4)
    ax.set_title("OOD generalization gap (smaller is better)",
                 fontsize=10, pad=6)
    ax.axhline(0, color="k", lw=0.8)
    for bar, gap in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width() / 2,
                gap + 0.008 * (1 if gap >= 0 else -1),
                f"{gap:+.3f}", ha="center",
                va="bottom" if gap >= 0 else "top",
                fontsize=6)
    if gaps:
        max_pos = max(max(gaps), 0.0)
        min_neg = min(min(gaps), 0.0)
        ax.set_ylim(min_neg - 0.04, max_pos + 0.04)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.15, dpi=150)
    plt.close(fig)


def peft_summary_bars(report: dict, out_path: Path) -> None:
    """Grouped bar chart: per (backbone, PEFT) show test vs bolivia IoU side-by-side.

    Method bars are colored from PEFT_COLORS; in-distribution test is a solid bar,
    Bolivia held-out is the same color with hatching so the eye groups them by method.
    """
    keys = sorted(report.get("peft", {}).keys())
    test_vals, bol_vals, labels, base_colors = [], [], [], []
    for key in keys:
        backbone, peft = parse_peft_key(key)
        sp = report["peft"][key]["splits"]
        if "test" not in sp or "bolivia" not in sp:
            continue
        test_vals.append(sp["test"]["iou"]["mean"])
        bol_vals.append(sp["bolivia"]["iou"]["mean"])
        labels.append(f"{BACKBONE_LABELS.get(backbone, backbone)}\n{PEFT_LABELS.get(peft, peft)}")
        base_colors.append(PEFT_COLORS.get(peft, "#444"))

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(5.2, 4.4), constrained_layout=True)
    ax.bar(x - 0.20, test_vals, width=0.38, color=base_colors,
           edgecolor="black", linewidth=0.5, label="Sen1F11 test")
    ax.bar(x + 0.20, bol_vals,  width=0.38, color=base_colors, alpha=0.55,
           edgecolor="black", linewidth=0.5, hatch="//", label="Bolivia")
    short = [l.replace("SAM 2 Hiera-B+\n", "H+ ").replace("SAM ViT-B\n", "ViT-B ")
             for l in labels]
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=35, ha="right", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylabel("IoU", fontsize=9, labelpad=4)
    ax.set_ylim(0, max(max(test_vals), max(bol_vals)) * 1.15 if test_vals else 1)
    ax.set_title("Per-method IoU: in-dist. (solid) vs. Bolivia (hatched)",
                 fontsize=10, pad=6)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.15, dpi=150)
    plt.close(fig)


def reliability_diagram(report: dict, out_path: Path) -> None:
    """Bin predicted probability and plot empirical accuracy per bin.

    Reads from runs/confidence_test.json if it exists.
    """
    src = Path("runs/confidence_test.json")
    if not src.exists():
        return
    data = json.loads(src.read_text())
    bins = data.get("reliability_bins", [])
    if not bins:
        return
    fig, ax = plt.subplots(figsize=(4.4, 4.4), constrained_layout=True)
    centers = [b["bin_center"] for b in bins]
    accs = [b["accuracy"] for b in bins]
    confs = [b["confidence"] for b in bins]
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Perfect calibration")
    ax.plot(confs, accs, "o-", color="#1f77b4", markersize=5,
            linewidth=1.2, label="Model (MC dropout, 20 passes)")
    ax.set_xlabel("Average predicted probability", fontsize=9, labelpad=4)
    ax.set_ylabel("Empirical accuracy", fontsize=9, labelpad=4)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_title(f"Reliability diagram (ECE = {data.get('ece', 0):.3f})",
                 fontsize=10, pad=6)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.15, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("runs/aggregate_results.json"))
    p.add_argument("--output", type=Path, default=Path("thesis/figs"))
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    report = json.loads(args.results.read_text())

    table_tex = render_results_table(report)
    (args.output / "results_table.tex").write_text(table_tex)
    print(f"[figs] wrote {args.output / 'results_table.tex'}")

    iou_scatter(report, args.output / "iou_vs_bolivia.pdf")
    print(f"[figs] wrote {args.output / 'iou_vs_bolivia.pdf'}")

    ood_gap_bars(report, args.output / "ood_gap.pdf")
    print(f"[figs] wrote {args.output / 'ood_gap.pdf'}")

    peft_summary_bars(report, args.output / "peft_summary.pdf")
    print(f"[figs] wrote {args.output / 'peft_summary.pdf'}")

    reliability_diagram(report, args.output / "reliability_diagram.pdf")
    if (args.output / "reliability_diagram.pdf").exists():
        print(f"[figs] wrote {args.output / 'reliability_diagram.pdf'}")
    else:
        print("[figs] skipped reliability_diagram (no confidence_test.json yet)")


if __name__ == "__main__":
    main()
