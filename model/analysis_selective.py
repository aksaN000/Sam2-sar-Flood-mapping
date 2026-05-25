"""Selective-prediction curve using MC dropout uncertainty.

For each (backbone, peft) seed=42 checkpoint, runs MC dropout (N passes) on
the OOD test splits and computes IoU as a function of the kept fraction:
at threshold t in [0, 1], abstain on the pixels with the top-(1-t) MC std,
re-compute IoU on the kept fraction.

Output: curve per (cell, split). One PDF (selective_prediction.pdf) plus a
machine-readable JSON.

Use after the main sweep; checkpoints must already exist under runs/.
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
from model.models.confidence import MCDropoutPredictor
from model.train import FloodLightningModule, load_cfg

KEEP_FRACTIONS = [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]


@torch.no_grad()
def selective_curve_one(ckpt_path: Path, cfg_path: Path, split: str,
                        sen1floods11_root: Path, pakistan2022_root: Path,
                        n_passes: int, device: str) -> list[dict]:
    cfg = load_cfg(cfg_path)
    cfg.dataset.sen1floods11_root = str(sen1floods11_root)
    module = FloodLightningModule.load_from_checkpoint(
        str(ckpt_path), cfg=cfg, map_location="cpu",
    ).to(device).eval()

    ds_root = str(pakistan2022_root) if split == "pakistan2022" else str(sen1floods11_root)
    try:
        ds = Sen1Floods11Dataset(root=ds_root, split=split,
                                 polarimetric_mode=cfg.dataset.polarimetric_mode)
    except FileNotFoundError:
        del module
        torch.cuda.empty_cache()
        return []

    predictor = MCDropoutPredictor(module.model, n_passes=n_passes).to(device)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    all_prob = []
    all_std = []
    all_label = []
    for batch in dl:
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        mean_prob, std_prob = predictor(x)
        all_prob.append(mean_prob.cpu())
        all_std.append(std_prob.cpu())
        all_label.append(y.cpu())
    del module, predictor
    torch.cuda.empty_cache()
    probs = torch.cat(all_prob, dim=0).numpy()          # (N, H, W)
    stds = torch.cat(all_std, dim=0).numpy()
    labels = torch.cat(all_label, dim=0).numpy()
    valid = labels != 255
    preds = (probs > 0.5).astype(np.uint8)

    # flatten valid pixels for global threshold
    flat_std = stds[valid].flatten()
    out: list[dict] = []
    for kf in KEEP_FRACTIONS:
        if kf >= 1.0:
            keep_mask = valid
        else:
            thresh = np.quantile(flat_std, kf)
            keep_mask = valid & (stds <= thresh)
        tp = int(((preds == 1) & (labels == 1) & keep_mask).sum())
        fp = int(((preds == 1) & (labels == 0) & keep_mask).sum())
        fn = int(((preds == 0) & (labels == 1) & keep_mask).sum())
        iou = tp / max(1, tp + fp + fn)
        out.append({"keep_fraction": kf, "iou": iou,
                    "n_kept": int(keep_mask.sum()),
                    "n_valid": int(valid.sum())})
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--sen1floods11-root", type=Path, default=Path("./data/sen1floods11"))
    p.add_argument("--pakistan2022-root", type=Path, default=Path("./data/pakistan-2022-chips"))
    p.add_argument("--output-dir", type=Path, default=Path("thesis/figs"))
    p.add_argument("--json-out", type=Path, default=Path("runs/selective_prediction.json"))
    p.add_argument("--n-passes", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--splits", nargs="+", default=["bolivia", "pakistan2022"])
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for run_dir in sorted(args.runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if not run_dir.name.endswith(f"_seed{args.seed}"):
            continue
        # Only the main 24-cell + stretch checkpoints are useful here
        if not any(run_dir.name.startswith(b + "_") for b in
                   ("sam_vit_b", "sam2_hiera_bp", "sam_vit_l", "sam_vit_h", "sam2_hiera_l")):
            continue
        best = sorted(run_dir.glob("best-*.ckpt"))
        last = run_dir / "last.ckpt"
        if best:
            ckpt = best[-1]
        elif last.exists():
            ckpt = last
        else:
            continue
        cfg = Path(__file__).parent / "configs" / f"{run_dir.name}.yaml"
        if not cfg.exists():
            continue
        print(f"[selective] {run_dir.name}")
        cell: dict[str, list] = {}
        for split in args.splits:
            try:
                curve = selective_curve_one(
                    ckpt, cfg, split,
                    args.sen1floods11_root, args.pakistan2022_root,
                    args.n_passes, args.device,
                )
            except Exception as e:
                print(f"  {split}: failed: {e}")
                continue
            cell[split] = curve
            if curve:
                print(f"  {split}: kept=1.0 IoU={curve[0]['iou']:.3f}  "
                      f"kept=0.5 IoU={curve[6]['iou']:.3f}")
        results[run_dir.name] = cell

    args.json_out.write_text(json.dumps(results, indent=2))
    print(f"[selective] wrote {args.json_out}")

    # Figure: one panel per split, one curve per cell. Shared legend on the
    # right so it does not overlap the data inside any panel.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        splits = args.splits
        backbone_labels = {
            "sam2_hiera_bp": "SAM 2 Hiera-B+", "sam2_hiera_l": "SAM 2 Hiera-L",
            "sam_vit_b": "SAM ViT-B", "sam_vit_l": "SAM ViT-L", "sam_vit_h": "SAM ViT-H",
        }
        peft_labels = {"lora": "LoRA", "dora": "DoRA",
                       "convlora": "Conv-LoRA", "adaptformer": "AdaptFormer"}
        split_titles = {"bolivia": "Bolivia (held-out country)",
                        "pakistan2022": "Pakistan-2022 (OOD geography + time)",
                        "test": "Sen1Floods11 test (in-distribution)",
                        "pakistan": "Pakistan (Sen1Floods11 subset)"}

        def pretty(name: str) -> str:
            parts = name.rsplit("_seed", 1)[0]
            for b in sorted(backbone_labels, key=len, reverse=True):
                if parts.startswith(b + "_"):
                    return f"{backbone_labels[b]} + {peft_labels.get(parts[len(b)+1:], parts[len(b)+1:])}"
            return parts

        fig, axes = plt.subplots(len(splits), 1,
                                 figsize=(5.0, 7.5),
                                 squeeze=False,
                                 gridspec_kw={"bottom": 0.13, "hspace": 0.32,
                                              "top": 0.96, "left": 0.12, "right": 0.97})
        handle_map: dict[str, object] = {}
        for ax, split in zip(axes[:, 0], splits):
            for name, cell in results.items():
                curve = cell.get(split, [])
                if not curve:
                    continue
                xs = [c["keep_fraction"] for c in curve]
                ys = [c["iou"] for c in curve]
                label = pretty(name)
                line, = ax.plot(xs, ys, marker="o", linewidth=1.4, markersize=4,
                                label=label)
                handle_map.setdefault(label, line)
            ax.set_xlabel("Kept fraction (lowest-uncertainty pixels)",
                          fontsize=10, labelpad=6)
            ax.set_ylabel("IoU on kept pixels", fontsize=10, labelpad=6)
            ax.set_title(split_titles.get(split, split), fontsize=11, pad=6)
            ax.tick_params(axis="both", labelsize=9)
            ax.invert_xaxis()
            ax.grid(True, alpha=0.3)
        fig.legend(list(handle_map.values()), list(handle_map.keys()),
                   loc="lower center", bbox_to_anchor=(0.5, 0.01),
                   fontsize=7, ncol=3, frameon=True, framealpha=0.95)
        fig.savefig(args.output_dir / "selective_prediction.pdf",
                    bbox_inches="tight", pad_inches=0.25, dpi=150)
        plt.close(fig)
        print(f"[selective] wrote {args.output_dir / 'selective_prediction.pdf'}")
    except Exception as e:
        print(f"[selective] figure skipped: {e}")


if __name__ == "__main__":
    main()
