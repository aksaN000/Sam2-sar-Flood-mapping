"""Deep ensemble + TTA + temperature-scaled evaluation.

For each (backbone, PEFT) cell, this script:
  1. Loads the three seed checkpoints,
  2. For each chip in each evaluation split, runs all three models, optionally with
     8 test-time-augmentation views (4 rotations x 2 flips), and averages the
     sigmoid probabilities,
  3. Optionally divides logits by a temperature T (fit on the valid split per cell)
     before sigmoid, for better calibration,
  4. Writes a JSON record with the ensemble IoU/F1/precision/recall/ECE per split.

Output: runs/ensemble_results.json (schema mirrors aggregate_results.json's `peft`).

This is a strictly post-hoc gain on top of existing trained checkpoints.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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


def find_best_ckpt(run_dir: Path) -> Path | None:
    best = sorted(run_dir.glob("best-*.ckpt"))
    if best:
        return best[-1]
    last = run_dir / "last.ckpt"
    return last if last.exists() else None


def load_module(ckpt_path: Path, cfg_path: Path, sen1floods11_root: Path,
                device: str) -> FloodLightningModule:
    cfg = load_cfg(cfg_path)
    cfg.dataset.sen1floods11_root = str(sen1floods11_root)
    module = FloodLightningModule.load_from_checkpoint(
        str(ckpt_path), cfg=cfg, map_location="cpu",
    ).to(device).eval()
    return module, cfg


def tta_forward(model: torch.nn.Module, x: torch.Tensor, use_tta: bool) -> torch.Tensor:
    """Average sigmoid probabilities over rotations + flips. Returns (B,H,W)."""
    if not use_tta:
        return torch.sigmoid(model(x))
    accum = None; n = 0
    for k in range(4):
        for flip in (False, True):
            xt = torch.rot90(x, k=k, dims=(-2, -1))
            if flip:
                xt = torch.flip(xt, dims=(-1,))
            logits = model(xt)
            probs = torch.sigmoid(logits)
            # invert flip + rotation
            if flip:
                probs = torch.flip(probs, dims=(-1,))
            probs = torch.rot90(probs, k=-k, dims=(-2, -1))
            accum = probs if accum is None else accum + probs
            n += 1
    return accum / n


def fit_temperature(probs_val: np.ndarray, labels_val: np.ndarray,
                    grid: np.ndarray | None = None) -> float:
    """Pick T in [0.1, 5.0] that minimizes ECE on the valid split."""
    if grid is None:
        grid = np.linspace(0.3, 3.0, 28)
    valid = labels_val != 255
    labels = labels_val[valid].astype(np.int32)
    probs = np.clip(probs_val[valid], 1e-6, 1 - 1e-6)
    best_T, best_ece = 1.0, 1.0
    for T in grid:
        # rescale: logit = log(p/(1-p)) / T; new_p = sigmoid(logit)
        logits = np.log(probs / (1 - probs)) / T
        p_T = 1.0 / (1.0 + np.exp(-logits))
        # quick ECE
        ece = 0.0
        edges = np.linspace(0, 1, 16)
        for i in range(15):
            m = (p_T >= edges[i]) & (p_T < edges[i + 1] if i < 14 else p_T <= edges[i + 1])
            n = int(m.sum())
            if n == 0: continue
            conf = float(p_T[m].mean())
            preds = (p_T[m] > 0.5).astype(np.int32)
            acc = float((preds == labels[m]).mean())
            ece += (n / len(labels)) * abs(conf - acc)
        if ece < best_ece:
            best_T, best_ece = float(T), float(ece)
    return best_T


@torch.no_grad()
def gather_probs_one_split(modules, ds, device, use_tta: bool) -> tuple[np.ndarray, np.ndarray]:
    """Returns (ensemble_probs (N*H*W,), labels (N*H*W,)) flattened."""
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    all_p, all_y = [], []
    for batch in dl:
        x = batch["image"].to(device)
        y = batch["label"].cpu().numpy().reshape(-1)
        per_member = [tta_forward(m.model, x, use_tta) for m in modules]
        ens = torch.stack(per_member, dim=0).mean(dim=0)  # (1, H, W)
        all_p.append(ens.cpu().numpy().reshape(-1))
        all_y.append(y)
    return np.concatenate(all_p), np.concatenate(all_y)


def compute_metrics_from_arrays(probs: np.ndarray, labels: np.ndarray,
                                 device: str) -> dict:
    valid = labels != 255
    preds = (probs > 0.5).astype(np.int32)
    pt = torch.from_numpy(preds[valid]).to(device)
    yt = torch.from_numpy(labels[valid].astype(np.int32)).to(device)
    iou = BinaryJaccardIndex(ignore_index=255).to(device)(pt, yt).item()
    f1 = BinaryF1Score(ignore_index=255).to(device)(pt, yt).item()
    pr = BinaryPrecision(ignore_index=255).to(device)(pt, yt).item()
    rc = BinaryRecall(ignore_index=255).to(device)(pt, yt).item()
    # ECE via a Python helper (probs are numpy)
    edges = np.linspace(0, 1, 16)
    ece = 0.0; total = int(valid.sum())
    p_v = probs[valid]; y_v = labels[valid].astype(np.int32)
    for i in range(15):
        m = (p_v >= edges[i]) & (p_v < edges[i + 1] if i < 14 else p_v <= edges[i + 1])
        n = int(m.sum())
        if n == 0: continue
        conf = float(p_v[m].mean())
        acc = float(((p_v[m] > 0.5).astype(np.int32) == y_v[m]).mean())
        ece += (n / total) * abs(conf - acc)
    return {"iou": float(iou), "f1": float(f1),
            "precision": float(pr), "recall": float(rc),
            "ece": float(ece), "n_chips": int(probs.size // (512 * 512))}


def apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    p = np.clip(probs, 1e-6, 1 - 1e-6)
    return 1.0 / (1.0 + np.exp(-np.log(p / (1 - p)) / T))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--output", type=Path, default=Path("runs/ensemble_results.json"))
    p.add_argument("--splits", nargs="+",
                   default=["test", "bolivia", "pakistan", "pakistan2022"])
    p.add_argument("--sen1floods11-root", type=Path, default=Path("./data/sen1floods11"))
    p.add_argument("--pakistan2022-root", type=Path, default=Path("./data/pakistan-2022-chips"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-tta", action="store_true", help="skip TTA (faster, no IoU gain)")
    p.add_argument("--no-temp", action="store_true", help="skip temperature scaling")
    args = p.parse_args()

    # Discover (backbone, peft) cells in the main + stretch + polari sections
    PEFT_RE = __import__("re").compile(
        r"^(?P<backbone>sam_vit_b|sam2_hiera_bp|sam_vit_l|sam_vit_h|sam2_hiera_l)_"
        r"(?P<peft>lora|dora|convlora|adaptformer)_seed(?P<seed>\d+)$"
    )
    groups = defaultdict(list)
    for d in sorted(args.runs_dir.iterdir()):
        if not d.is_dir(): continue
        m = PEFT_RE.match(d.name)
        if not m: continue
        ckpt = find_best_ckpt(d)
        if ckpt is None: continue
        groups[(m["backbone"], m["peft"])].append((int(m["seed"]), ckpt, d.name))

    print(f"[ensemble] {len(groups)} (backbone, peft) groups to process")
    use_tta = not args.no_tta
    use_temp = not args.no_temp

    results: dict[str, dict] = {}
    for (backbone, peft), seed_list in sorted(groups.items()):
        if len(seed_list) < 2:
            print(f"  SKIP {backbone}__{peft} ({len(seed_list)} seeds, need >=2)")
            continue
        key = f"{backbone}__{peft}"
        print(f"  ENSEMBLE {key} ({len(seed_list)} seeds, tta={use_tta}, temp={use_temp})")
        seed_list.sort()
        cfg_path = Path(__file__).parent / "configs" / f"{seed_list[0][2]}.yaml"
        modules = []
        for seed, ckpt, name in seed_list:
            cfg_p = Path(__file__).parent / "configs" / f"{name}.yaml"
            module, cfg = load_module(ckpt, cfg_p, args.sen1floods11_root, args.device)
            modules.append(module)

        # Fit temperature on the valid split
        T = 1.0
        if use_temp:
            try:
                ds_val = Sen1Floods11Dataset(root=str(args.sen1floods11_root),
                                             split="valid",
                                             polarimetric_mode=cfg.dataset.polarimetric_mode)
                p_val, y_val = gather_probs_one_split(modules, ds_val, args.device, use_tta=False)
                T = fit_temperature(p_val, y_val)
                print(f"    fitted T = {T:.3f}")
            except FileNotFoundError as e:
                print(f"    temperature skip: {e}")

        cell = {"label": key, "n_seeds": len(seed_list), "temperature": T,
                "tta": use_tta, "splits": {}}
        for split in args.splits:
            ds_root = (args.pakistan2022_root if split == "pakistan2022"
                       else args.sen1floods11_root)
            try:
                ds = Sen1Floods11Dataset(root=str(ds_root), split=split,
                                         polarimetric_mode=cfg.dataset.polarimetric_mode)
            except FileNotFoundError as e:
                print(f"    {split}: skipped ({e})")
                continue
            probs, labels = gather_probs_one_split(modules, ds, args.device, use_tta)
            if use_temp and T != 1.0:
                probs = apply_temperature(probs, T)
            m = compute_metrics_from_arrays(probs, labels, args.device)
            cell["splits"][split] = m
            print(f"    {split}: IoU={m['iou']:.3f} F1={m['f1']:.3f} ECE={m['ece']:.3f}")
        results[key] = cell

        for module in modules:
            del module
        torch.cuda.empty_cache()

    args.output.write_text(json.dumps(results, indent=2))
    print(f"[ensemble] wrote {args.output}")


if __name__ == "__main__":
    main()
