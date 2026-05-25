"""Aggregate trained checkpoints across seeds and emit a single results JSON.

Walks `runs/` for run directories matching the pattern
    {backbone}_{peft}_seed{N}/
groups checkpoints by (backbone, peft), evaluates each on the test and
bolivia splits, then writes mean / std / per-seed numbers to a single
JSON file. Also picks up U-Net runs (unet_seed{N}/) and the zero-shot
baseline if present in runs/zeroshot_results.json.

The output schema matches what `make_figures.py` expects, so a single
file feeds both the figures and the thesis tables.

CLI
---
    python -m model.aggregate_results --runs-dir runs --output runs/aggregate_results.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.eval import evaluate_one
from model.train_unet import UNetModule  # ensures import path for U-Net ckpts


# Regex captures backbone, peft, seed for PEFT sweep run dir names.
PEFT_RUN_RE = re.compile(
    r"^(?P<backbone>sam_vit_b|sam2_hiera_bp)_(?P<peft>lora|dora|convlora|adaptformer)_seed(?P<seed>\d+)$"
)
STRETCH_RUN_RE = re.compile(
    r"^(?P<backbone>sam_vit_l|sam_vit_h|sam2_hiera_l)_(?P<peft>lora|dora|convlora|adaptformer)_seed(?P<seed>\d+)$"
)
POLARI_RUN_RE = re.compile(
    r"^polari_(?P<polari>ratio|diff|single)_sam2_(?P<peft>lora|dora|convlora|adaptformer)_seed(?P<seed>\d+)$"
)
LINEARPROBE_RUN_RE = re.compile(
    r"^extra_linearprobe_(?P<backbone>sam_vit_b|sam2_hiera_bp)_seed(?P<seed>\d+)$"
)
EXTRA_POLARI_RUN_RE = re.compile(
    r"^extra_polari_(?P<polari>diff|single)_sam2_(?P<peft>lora|dora|convlora|adaptformer)_seed(?P<seed>\d+)$"
)
UNET_RUN_RE = re.compile(r"^unet_seed(?P<seed>\d+)$")


def find_runs(runs_dir: Path) -> dict:
    """Return classified run directories.

    Returns a dict with keys:
        peft     : {(backbone, peft): [(seed, ckpt), ...]}      24-cell main sweep
        stretch  : {(backbone, peft): [(seed, ckpt), ...]}      9 large-backbone configs
        polari   : {(polari, peft):   [(seed, ckpt), ...]}      polarimetric mode ablation
        linearprobe : {backbone: [(seed, ckpt), ...]}            decoder-only baseline
        polari_pi: {(polari, peft):  [(seed, ckpt), ...]}        polari x PEFT interaction
        unet     : [(seed, ckpt), ...]                           U-Net baseline
    """
    peft_groups: dict[tuple[str, str], list[tuple[int, Path]]] = defaultdict(list)
    stretch_groups: dict[tuple[str, str], list[tuple[int, Path]]] = defaultdict(list)
    polari_groups: dict[tuple[str, str], list[tuple[int, Path]]] = defaultdict(list)
    linearprobe_groups: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    polari_pi_groups: dict[tuple[str, str], list[tuple[int, Path]]] = defaultdict(list)
    unet_seeds: list[tuple[int, Path]] = []

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        # Prefer the best-val-IoU checkpoint when present; fall back to last.ckpt.
        # best-*.ckpt is the Lightning ModelCheckpoint output with monitor=val/iou
        # save_top_k=1, so the model state at the peak validation epoch.
        best_ckpts = sorted(run_dir.glob("best-*.ckpt"))
        ckpt = best_ckpts[-1] if best_ckpts else (run_dir / "last.ckpt")
        if not ckpt.exists():
            continue

        if (m := PEFT_RUN_RE.match(run_dir.name)):
            peft_groups[(m["backbone"], m["peft"])].append((int(m["seed"]), ckpt))
        elif (m := STRETCH_RUN_RE.match(run_dir.name)):
            stretch_groups[(m["backbone"], m["peft"])].append((int(m["seed"]), ckpt))
        elif (m := POLARI_RUN_RE.match(run_dir.name)):
            polari_groups[(m["polari"], m["peft"])].append((int(m["seed"]), ckpt))
        elif (m := LINEARPROBE_RUN_RE.match(run_dir.name)):
            linearprobe_groups[m["backbone"]].append((int(m["seed"]), ckpt))
        elif (m := EXTRA_POLARI_RUN_RE.match(run_dir.name)):
            polari_pi_groups[(m["polari"], m["peft"])].append((int(m["seed"]), ckpt))
        elif (m := UNET_RUN_RE.match(run_dir.name)):
            unet_seeds.append((int(m["seed"]), ckpt))

    return {
        "peft": peft_groups,
        "stretch": stretch_groups,
        "polari": polari_groups,
        "linearprobe": linearprobe_groups,
        "polari_pi": polari_pi_groups,
        "unet": unet_seeds,
    }


def agg_metrics(per_seed_results: list[dict]) -> dict:
    """Aggregate IoU / F1 / precision / recall / ECE across seeds."""
    out = {"per_seed": per_seed_results, "n_seeds": len(per_seed_results)}
    for k in ("iou", "f1", "precision", "recall", "ece"):
        vals = [r[k] for r in per_seed_results if k in r]
        if not vals:
            continue
        out[k] = {"mean": mean(vals), "std": stdev(vals) if len(vals) > 1 else 0.0}
    return out


def evaluate_group(
    label: str,
    seed_ckpts: list[tuple[int, Path]],
    splits: list[str],
    sen1floods11_root: Path,
    pakistan2022_root: Path | None = None,
) -> dict:
    """Evaluate each (seed, ckpt) on each split; aggregate per split."""
    out = {"label": label, "splits": {}}
    for split in splits:
        per_seed = []
        for seed, ckpt in seed_ckpts:
            print(f"  [{label}] seed={seed} split={split} ckpt={ckpt}")
            try:
                r = evaluate_one(
                    ckpt, split, sen1floods11_root, None, batch_size=1,
                    pakistan2022_root=pakistan2022_root,
                )
            except FileNotFoundError as e:
                print(f"    skipped: {e}")
                continue
            r["seed"] = seed
            per_seed.append(r)
        if per_seed:
            out["splits"][split] = agg_metrics(per_seed)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--output", type=Path, default=Path("runs/aggregate_results.json"))
    p.add_argument("--splits", nargs="+",
                   default=["test", "bolivia", "pakistan", "pakistan2022"])
    p.add_argument("--sen1floods11-root", type=Path,
                   default=Path("./data/sen1floods11"))
    p.add_argument("--pakistan2022-root", type=Path,
                   default=Path("./data/pakistan-2022-chips"))
    args = p.parse_args()

    runs = find_runs(args.runs_dir)
    print(f"[aggregate] found "
          f"{len(runs['peft'])} main, "
          f"{len(runs['stretch'])} stretch, "
          f"{len(runs['polari'])} polari, "
          f"{len(runs['linearprobe'])} linear-probe, "
          f"{len(runs['polari_pi'])} polari x PEFT, "
          f"{len(runs['unet'])} U-Net seeds")

    report = {
        "splits": args.splits,
        "peft": {},
        "stretch": {},
        "polari": {},
        "linearprobe": {},
        "polari_pi": {},
        "unet": None,
        "zeroshot": None,
    }

    def eval_group_dict(groups, label_fmt):
        out = {}
        for k, seed_ckpts in sorted(groups.items()):
            seed_ckpts.sort()
            key = label_fmt(k) if not isinstance(k, str) else k
            out[key] = evaluate_group(
                key, seed_ckpts, args.splits, args.sen1floods11_root,
                pakistan2022_root=args.pakistan2022_root,
            )
        return out

    report["peft"]        = eval_group_dict(runs["peft"],        lambda k: f"{k[0]}__{k[1]}")
    report["stretch"]     = eval_group_dict(runs["stretch"],     lambda k: f"{k[0]}__{k[1]}")
    report["polari"]      = eval_group_dict(runs["polari"],      lambda k: f"{k[0]}__{k[1]}")
    report["linearprobe"] = eval_group_dict(runs["linearprobe"], lambda k: k)
    report["polari_pi"]   = eval_group_dict(runs["polari_pi"],   lambda k: f"{k[0]}__{k[1]}")

    if runs["unet"]:
        runs["unet"].sort()
        report["unet"] = evaluate_group(
            "unet", runs["unet"], args.splits, args.sen1floods11_root,
            pakistan2022_root=args.pakistan2022_root,
        )

    zeroshot_path = args.runs_dir / "zeroshot_results.json"
    if zeroshot_path.exists():
        report["zeroshot"] = json.loads(zeroshot_path.read_text())

    args.output.write_text(json.dumps(report, indent=2))
    print(f"[aggregate] wrote {args.output}")
    print(json.dumps({
        "peft_keys": list(report["peft"].keys()),
        "stretch_keys": list(report["stretch"].keys()),
        "polari_keys": list(report["polari"].keys()),
        "linearprobe_keys": list(report["linearprobe"].keys()),
        "polari_pi_keys": list(report["polari_pi"].keys()),
        "unet_present": report["unet"] is not None,
        "zeroshot_present": report["zeroshot"] is not None,
    }, indent=2))


if __name__ == "__main__":
    main()
