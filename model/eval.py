"""Evaluation entry point for trained SAM/SAM 2 SAR flood adapters.

Loads one or more checkpoints, runs inference on a chosen Sen1Floods11
split, and writes IoU / F1 / precision / recall / ECE to a JSON report.
Multiple checkpoints (typically the three thesis seeds 42, 123, 20025)
are aggregated as mean and standard deviation.

CLI
---
    python -m model.eval --checkpoint runs/sam2_hiera_bp_lora_seed42/last.ckpt \\
                         --split test --polarimetric-mode ratio

    # Aggregate across seeds:
    python -m model.eval \\
        --checkpoint runs/sam2_hiera_bp_lora_seed42/last.ckpt \\
                     runs/sam2_hiera_bp_lora_seed123/last.ckpt \\
                     runs/sam2_hiera_bp_lora_seed20025/last.ckpt \\
        --split bolivia
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryJaccardIndex,
    BinaryPrecision,
    BinaryRecall,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.datasets.sen1floods11 import Sen1Floods11Dataset
from model.models.confidence import expected_calibration_error
from model.train import FloodLightningModule, load_cfg


def find_cfg_for_ckpt(ckpt_path: Path) -> Path:
    """A checkpoint's run dir name matches `model/configs/<name>.yaml`."""
    run_name = ckpt_path.parent.name
    return Path(__file__).parent / "configs" / f"{run_name}.yaml"


def _is_unet_run(ckpt_path: Path) -> bool:
    return ckpt_path.parent.name.startswith("unet_seed")


def evaluate_one(
    ckpt_path: Path,
    split: str,
    sen1floods11_root: Path | None,
    polarimetric_mode: str | None,
    batch_size: int,
    pakistan2022_root: Path | None = None,
) -> dict:
    cfg_path = find_cfg_for_ckpt(ckpt_path)
    cfg = load_cfg(cfg_path)
    if sen1floods11_root is not None:
        cfg.dataset.sen1floods11_root = str(sen1floods11_root)
    if polarimetric_mode is not None:
        cfg.dataset.polarimetric_mode = polarimetric_mode

    if _is_unet_run(ckpt_path):
        from model.train_unet import UNetModule
        module = UNetModule.load_from_checkpoint(
            str(ckpt_path), cfg=cfg, map_location="cpu"
        )
    else:
        module = FloodLightningModule.load_from_checkpoint(
            str(ckpt_path), cfg=cfg, map_location="cpu"
        )
    module.eval()
    if torch.cuda.is_available():
        module = module.cuda()

    # The pakistan2022 split lives at a separate chip root (the TU Wien-derived
    # chips built by data_pipelines/pakistan_2022/acquire.py).
    if split == "pakistan2022":
        if pakistan2022_root is None:
            pakistan2022_root = Path("./data/pakistan-2022-chips")
        ds_root = str(pakistan2022_root)
    else:
        ds_root = cfg.dataset.sen1floods11_root

    ds = Sen1Floods11Dataset(
        root=ds_root,
        split=split,
        polarimetric_mode=cfg.dataset.polarimetric_mode,
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    iou_m = BinaryJaccardIndex(ignore_index=255)
    f1_m = BinaryF1Score(ignore_index=255)
    pr_m = BinaryPrecision(ignore_index=255)
    rc_m = BinaryRecall(ignore_index=255)
    if torch.cuda.is_available():
        iou_m = iou_m.cuda(); f1_m = f1_m.cuda(); pr_m = pr_m.cuda(); rc_m = rc_m.cuda()

    ece_total = 0.0
    ece_count = 0
    with torch.no_grad():
        for batch in dl:
            images = batch["image"].to(module.device, non_blocking=True)
            labels = batch["label"].to(module.device, non_blocking=True)
            logits = module(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()
            iou_m.update(preds, labels.int())
            f1_m.update(preds, labels.int())
            pr_m.update(preds, labels.int())
            rc_m.update(preds, labels.int())
            ece_total += expected_calibration_error(probs, labels) * images.size(0)
            ece_count += images.size(0)

    return {
        "iou": float(iou_m.compute().item()),
        "f1": float(f1_m.compute().item()),
        "precision": float(pr_m.compute().item()),
        "recall": float(rc_m.compute().item()),
        "ece": ece_total / max(1, ece_count),
        "n_chips": ece_count,
    }


def aggregate(per_seed: list[dict]) -> dict:
    """Return mean and standard deviation across seeds."""
    import statistics
    keys = ["iou", "f1", "precision", "recall", "ece"]
    out = {}
    for k in keys:
        vals = [r[k] for r in per_seed]
        out[k] = {
            "mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "per_seed": vals,
        }
    out["n_chips"] = per_seed[0]["n_chips"] if per_seed else 0
    out["n_seeds"] = len(per_seed)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, nargs="+", required=True,
                   help="One or more checkpoint paths to aggregate across.")
    p.add_argument("--split",
                   choices=["train", "valid", "test", "bolivia", "pakistan", "pakistan2022"],
                   required=True)
    p.add_argument("--sen1floods11-root", type=Path, default=None)
    p.add_argument("--pakistan2022-root", type=Path,
                   default=Path("./data/pakistan-2022-chips"))
    p.add_argument("--polarimetric-mode", choices=["ratio", "diff", "single"], default=None)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    per_seed = []
    for ckpt in args.checkpoint:
        print(f"[eval] {ckpt} -> split={args.split}")
        result = evaluate_one(
            ckpt, args.split, args.sen1floods11_root, args.polarimetric_mode, args.batch_size,
            pakistan2022_root=args.pakistan2022_root,
        )
        print(f"        IoU={result['iou']:.4f} F1={result['f1']:.4f} "
              f"P={result['precision']:.4f} R={result['recall']:.4f} ECE={result['ece']:.4f}")
        per_seed.append(result)
    agg = aggregate(per_seed)
    report = {"split": args.split, "checkpoints": [str(c) for c in args.checkpoint], **agg}
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[eval] wrote {args.output}")
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
