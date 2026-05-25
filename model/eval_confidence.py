"""Produce runs/confidence_test.json for the reliability diagram.

Runs Monte Carlo dropout on a chosen trained adapter checkpoint, bins the
predicted probabilities into N bins, computes the empirical accuracy and
average confidence per bin, and writes them to JSON in the schema that
`model/make_figures.py:reliability_diagram` expects.

Default config: SAM 2 Hiera-Base-Plus + AdaptFormer, seed 42 (the winning
configuration on Sen1Floods11). Default split: test (90 chips).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.datasets.sen1floods11 import Sen1Floods11Dataset
from model.models.confidence import MCDropoutPredictor
from model.train import FloodLightningModule, load_cfg


@torch.no_grad()
def gather_probs_labels(ckpt_path: Path, cfg_path: Path, split: str,
                        sen1floods11_root: Path, pakistan2022_root: Path,
                        n_passes: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    cfg = load_cfg(cfg_path)
    cfg.dataset.sen1floods11_root = str(sen1floods11_root)
    module = FloodLightningModule.load_from_checkpoint(
        str(ckpt_path), cfg=cfg, map_location="cpu",
    ).to(device).eval()
    ds_root = str(pakistan2022_root) if split == "pakistan2022" else str(sen1floods11_root)
    ds = Sen1Floods11Dataset(root=ds_root, split=split,
                             polarimetric_mode=cfg.dataset.polarimetric_mode)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    predictor = MCDropoutPredictor(module.model, n_passes=n_passes).to(device)
    all_probs, all_labels = [], []
    for batch in dl:
        x = batch["image"].to(device)
        y = batch["label"].cpu().numpy()
        mean_prob, _ = predictor(x)
        all_probs.append(mean_prob.cpu().numpy().reshape(-1))
        all_labels.append(y.reshape(-1))
    del module, predictor
    torch.cuda.empty_cache()
    return np.concatenate(all_probs), np.concatenate(all_labels)


def reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> tuple[list[dict], float]:
    """Equal-width binning. Returns per-bin records + scalar ECE."""
    valid = labels != 255
    probs = probs[valid]
    labels = labels[valid].astype(np.int32)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[dict] = []
    ece = 0.0
    total = len(probs)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        n = int(mask.sum())
        if n == 0:
            out.append({"bin_center": float((lo + hi) / 2),
                        "n": 0, "confidence": 0.0, "accuracy": 0.0})
            continue
        conf = float(probs[mask].mean())
        preds = (probs[mask] > 0.5).astype(np.int32)
        acc = float((preds == labels[mask]).mean())
        out.append({"bin_center": float((lo + hi) / 2),
                    "n": n, "confidence": conf, "accuracy": acc})
        ece += (n / total) * abs(conf - acc)
    return out, float(ece)


def find_best_ckpt(run_dir: Path) -> Path | None:
    best_glob = sorted(run_dir.glob("best-*.ckpt"))
    if best_glob:
        return best_glob[-1]
    last = run_dir / "last.ckpt"
    return last if last.exists() else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path,
                   default=Path("runs/sam2_hiera_bp_adaptformer_seed42"))
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--split", choices=["test", "bolivia", "pakistan", "pakistan2022"],
                   default="test")
    p.add_argument("--n-passes", type=int, default=20)
    p.add_argument("--n-bins", type=int, default=15)
    p.add_argument("--sen1floods11-root", type=Path, default=Path("./data/sen1floods11"))
    p.add_argument("--pakistan2022-root", type=Path, default=Path("./data/pakistan-2022-chips"))
    p.add_argument("--output", type=Path, default=Path("runs/confidence_test.json"))
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    if args.ckpt is None:
        args.ckpt = find_best_ckpt(args.run_dir)
        if args.ckpt is None:
            raise SystemExit(f"No ckpt in {args.run_dir}")
    if args.config is None:
        args.config = Path(__file__).parent / "configs" / f"{args.run_dir.name}.yaml"

    print(f"[confidence] ckpt={args.ckpt}")
    print(f"[confidence] config={args.config}")
    print(f"[confidence] split={args.split} n_passes={args.n_passes} n_bins={args.n_bins}")

    probs, labels = gather_probs_labels(
        args.ckpt, args.config, args.split,
        args.sen1floods11_root, args.pakistan2022_root,
        args.n_passes, args.device,
    )
    bins, ece = reliability_bins(probs, labels, n_bins=args.n_bins)
    out = {
        "ckpt": str(args.ckpt),
        "config": str(args.config),
        "split": args.split,
        "n_passes": args.n_passes,
        "n_bins": args.n_bins,
        "n_pixels": int(len(probs)),
        "n_valid_pixels": int((labels != 255).sum()),
        "ece": ece,
        "reliability_bins": bins,
    }
    args.output.write_text(json.dumps(out, indent=2))
    print(f"[confidence] ECE = {ece:.4f}")
    print(f"[confidence] wrote {args.output}")


if __name__ == "__main__":
    main()
