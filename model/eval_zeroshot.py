"""Zero-shot SAM and SAM 2 baseline evaluation.

Loads the pretrained backbone without any fine-tuning, prompts it with
a whole-image box covering the chip, and records IoU / F1 / precision /
recall on the requested Sen1Floods11 split(s). This is the unstrained
foundation-model floor that the PEFT methods must improve on.

CLI
---
    python -m model.eval_zeroshot --backbone sam-vit-b --split test
    python -m model.eval_zeroshot --backbone sam2-hiera-bp --split test --split bolivia
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
from model.models.sam_adapter import SAMAdapter
from model.models.sam2_adapter import SAM2Adapter


def build_zeroshot(backbone: str) -> torch.nn.Module:
    """Return a frozen adapter where no PEFT delta is trained.

    We reuse SAMAdapter / SAM2Adapter with peft_method='lora' and rank=8
    so the wrapper shape matches what the PEFT comparison uses, but we
    never train. The LoRA deltas are zero-initialized for B (the up
    projection), so the initial forward equals the base linear -- i.e.,
    pure zero-shot behavior.
    """
    if backbone == "sam-vit-b":
        return SAMAdapter(peft_method="lora", rank=8, mask_decoder_strategy="lora-r4")
    if backbone == "sam2-hiera-bp":
        return SAM2Adapter(peft_method="lora", rank=8, mask_decoder_strategy="lora-r4")
    raise ValueError(f"Unknown backbone {backbone!r}")


def evaluate(model: torch.nn.Module, root: str, split: str,
             polarimetric_mode: str = "ratio",
             pakistan2022_root: str | None = None) -> dict:
    # The pakistan2022 split lives at a separate chip root.
    effective_root = pakistan2022_root if split == "pakistan2022" and pakistan2022_root else root
    ds = Sen1Floods11Dataset(root=effective_root, split=split, polarimetric_mode=polarimetric_mode)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    device = next(model.parameters()).device
    iou = BinaryJaccardIndex(ignore_index=255).to(device)
    f1 = BinaryF1Score(ignore_index=255).to(device)
    pr = BinaryPrecision(ignore_index=255).to(device)
    rc = BinaryRecall(ignore_index=255).to(device)
    ece_total, n = 0.0, 0
    model.eval()
    with torch.no_grad():
        for batch in dl:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()
            iou.update(preds, labels.int())
            f1.update(preds, labels.int())
            pr.update(preds, labels.int())
            rc.update(preds, labels.int())
            ece_total += expected_calibration_error(probs, labels) * images.size(0)
            n += images.size(0)
    return {
        "iou": float(iou.compute().item()),
        "f1": float(f1.compute().item()),
        "precision": float(pr.compute().item()),
        "recall": float(rc.compute().item()),
        "ece": ece_total / max(1, n),
        "n_chips": n,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", choices=["sam-vit-b", "sam2-hiera-bp"], required=True)
    p.add_argument("--split",
                   choices=["train", "valid", "test", "bolivia", "pakistan", "pakistan2022"],
                   action="append", required=True,
                   help="Pass once per split to evaluate; can be repeated.")
    p.add_argument("--sen1floods11-root", type=Path,
                   default=Path("./data/sen1floods11"))
    p.add_argument("--pakistan2022-root", type=Path,
                   default=Path("./data/pakistan-2022-chips"))
    p.add_argument("--polarimetric-mode", choices=["ratio", "diff", "single"],
                   default="ratio")
    p.add_argument("--output", type=Path,
                   default=Path("runs/zeroshot_results.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[zeroshot] loading {args.backbone}...")
    model = build_zeroshot(args.backbone)
    if torch.cuda.is_available():
        model = model.cuda()
    print(f"[zeroshot] running on {next(model.parameters()).device}")

    report = {"backbone": args.backbone, "polarimetric_mode": args.polarimetric_mode, "splits": {}}
    for split in args.split:
        print(f"[zeroshot] split={split}")
        try:
            result = evaluate(
                model, str(args.sen1floods11_root), split, args.polarimetric_mode,
                pakistan2022_root=str(args.pakistan2022_root),
            )
        except FileNotFoundError as e:
            print(f"           skipped: {e}")
            continue
        print(f"           IoU={result['iou']:.4f} F1={result['f1']:.4f} "
              f"P={result['precision']:.4f} R={result['recall']:.4f} ECE={result['ece']:.4f}")
        report["splits"][split] = result

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Merge with any existing zero-shot results so that running this script
    # multiple times (e.g. once for test+bolivia, then again for pakistan+
    # pakistan2022) accumulates splits instead of clobbering them.
    existing = {}
    if args.output.exists():
        try:
            existing = json.loads(args.output.read_text())
        except Exception:
            existing = {}
    backbone_entry = existing.get(args.backbone, {"backbone": args.backbone,
                                                   "polarimetric_mode": args.polarimetric_mode,
                                                   "splits": {}})
    backbone_entry["polarimetric_mode"] = args.polarimetric_mode
    backbone_entry.setdefault("splits", {})
    backbone_entry["splits"].update(report["splits"])
    existing[args.backbone] = backbone_entry
    with open(args.output, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"[zeroshot] wrote {args.output} (backbone {args.backbone} now has "
          f"splits: {sorted(backbone_entry['splits'].keys())})")


if __name__ == "__main__":
    main()
