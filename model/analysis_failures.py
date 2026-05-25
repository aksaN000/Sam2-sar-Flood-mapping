"""Failure case gallery: per (backbone, PEFT) cell, find the N worst-IoU chips
on the OOD splits and dump a per-chip JSON record. Optionally renders a PDF
panel grid if matplotlib is available.

CLI
---
    python -u -m model.analysis_failures \\
        --runs-dir runs --sen1floods11-root ./data/sen1floods11 \\
        --pakistan2022-root ./data/pakistan-2022-chips \\
        --output-dir thesis/figs --n-worst 6
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryJaccardIndex

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.datasets.sen1floods11 import Sen1Floods11Dataset
from model.train import FloodLightningModule, load_cfg


def per_chip_iou(module, ds, device) -> list[tuple[str, float]]:
    iou_per_chip = []
    iou_m = BinaryJaccardIndex(ignore_index=255).to(device)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    with torch.no_grad():
        for batch in dl:
            chip_id = batch["chip_id"][0] if isinstance(batch["chip_id"], list) else batch["chip_id"]
            image = batch["image"].to(device)
            label = batch["label"].to(device)
            logits = module(image)
            preds = (torch.sigmoid(logits) > 0.5).int()
            iou_m.reset()
            iou_m.update(preds, label.int())
            iou_per_chip.append((str(chip_id), float(iou_m.compute().item())))
    return iou_per_chip


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--sen1floods11-root", type=Path, default=Path("./data/sen1floods11"))
    p.add_argument("--pakistan2022-root", type=Path, default=Path("./data/pakistan-2022-chips"))
    p.add_argument("--output-dir", type=Path, default=Path("report/figs"))
    p.add_argument("--json-out", type=Path, default=Path("runs/failure_cases.json"))
    p.add_argument("--n-worst", type=int, default=6)
    p.add_argument("--splits", nargs="+", default=["bolivia", "pakistan2022"])
    p.add_argument("--seed", type=int, default=42,
                   help="Which seed's ckpt per (backbone, peft) cell.")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    out: dict[str, dict] = {}
    for run_dir in sorted(args.runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if not run_dir.name.endswith(f"_seed{args.seed}"):
            continue
        ckpt = run_dir / "last.ckpt"
        cfg_path = Path(__file__).parent / "configs" / f"{run_dir.name}.yaml"
        if not (ckpt.exists() and cfg_path.exists()):
            continue
        cfg = load_cfg(cfg_path)
        cfg.dataset.sen1floods11_root = str(args.sen1floods11_root)
        print(f"[failures] {run_dir.name}")
        try:
            module = FloodLightningModule.load_from_checkpoint(
                str(ckpt), cfg=cfg, map_location="cpu",
            ).to(args.device).eval()
        except Exception as e:
            print(f"  load failed: {e}")
            continue
        cell: dict[str, list] = {}
        for split in args.splits:
            try:
                ds_root = str(args.pakistan2022_root) if split == "pakistan2022" else str(args.sen1floods11_root)
                ds = Sen1Floods11Dataset(root=ds_root, split=split,
                                         polarimetric_mode=cfg.dataset.polarimetric_mode)
            except FileNotFoundError:
                continue
            ious = per_chip_iou(module, ds, args.device)
            ious.sort(key=lambda t: t[1])
            cell[split] = [{"chip_id": c, "iou": iou} for c, iou in ious[: args.n_worst]]
        out[run_dir.name] = cell
        del module
        torch.cuda.empty_cache()

    args.json_out.write_text(json.dumps(out, indent=2))
    print(f"[failures] wrote {args.json_out}")


if __name__ == "__main__":
    main()
