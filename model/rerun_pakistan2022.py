"""Re-evaluate Pakistan-2022 only on existing checkpoints, then patch aggregate.

Targeted rerun after fixing the Pakistan-2022 acquisition pipeline
(future_work/pakistan_2022/acquire_v2.py). This script does NOT retrain:
it walks runs/<config>/ directories, loads each best/last checkpoint,
runs inference on the v2 Pakistan-2022 chips, and patches the
`splits.pakistan2022` block of runs/aggregate_results.json in-place.

Usage:
    # 1. Make sure runs/ has the original checkpoints downloaded from GDrive.
    # 2. Make sure data/pakistan-2022-chips-v2/ has the v2 chip set.
    # 3. Run:
    python -m model.rerun_pakistan2022 \\
        --runs-dir runs \\
        --pakistan2022-root data/pakistan-2022-chips-v2 \\
        --aggregate runs/aggregate_results.json

The script writes a backup at runs/aggregate_results.before_v2.json before
patching, so the original numbers are recoverable.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.eval import evaluate_one  # type: ignore


def find_checkpoint(run_dir: Path) -> Path | None:
    """Prefer best-*.ckpt; fall back to last.ckpt."""
    best = sorted(run_dir.glob("best-*.ckpt"))
    if best:
        return best[-1]
    last = run_dir / "last.ckpt"
    return last if last.exists() else None


def parse_peft_key(run_name: str) -> tuple[str, str] | None:
    """Extract (backbone, peft) from a run directory name like
    sam2_hiera_bp_adaptformer_seed42 -> ('sam2_hiera_bp', 'adaptformer').
    Returns None for runs that do not match the (backbone, peft) pattern.
    """
    for b in ("sam2_hiera_bp", "sam2_hiera_l",
              "sam_vit_b", "sam_vit_l", "sam_vit_h"):
        if run_name.startswith(b + "_"):
            rest = run_name[len(b) + 1:]
            seed_idx = rest.rfind("_seed")
            if seed_idx == -1:
                return None
            peft = rest[:seed_idx]
            return b, peft
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--pakistan2022-root", type=Path,
                   default=Path("data/pakistan-2022-chips-v2"))
    p.add_argument("--aggregate", type=Path,
                   default=Path("runs/aggregate_results.json"))
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--dry-run", action="store_true",
                   help="List configs that would be re-evaluated without running.")
    args = p.parse_args()

    if not args.aggregate.exists():
        sys.exit(f"aggregate not found: {args.aggregate}")
    if not args.pakistan2022_root.exists():
        sys.exit(f"Pakistan-2022 v2 chip root not found: {args.pakistan2022_root}")

    # Backup aggregate before patching.
    backup = args.aggregate.with_name(
        args.aggregate.stem + ".before_v2.json"
    )
    if not backup.exists():
        shutil.copy(args.aggregate, backup)
        print(f"[rerun] backed up {args.aggregate} -> {backup}")

    aggregate = json.loads(args.aggregate.read_text())

    # Group runs by (backbone, peft) so we can average across seeds.
    cells: dict[tuple[str, str], list[Path]] = {}
    for run_dir in sorted(args.runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if run_dir.name.startswith("extra_") or run_dir.name.startswith("polari_"):
            continue
        key = parse_peft_key(run_dir.name)
        if key is None:
            continue
        cells.setdefault(key, []).append(run_dir)

    print(f"[rerun] {len(cells)} (backbone, PEFT) cells to re-evaluate")
    if args.dry_run:
        for (bb, peft), runs in sorted(cells.items()):
            print(f"  {bb} + {peft}: {len(runs)} seed runs")
        return

    for (bb, peft), seed_runs in sorted(cells.items()):
        ious, eces = [], []
        for run_dir in seed_runs:
            ckpt = find_checkpoint(run_dir)
            if ckpt is None:
                print(f"  SKIP {run_dir.name}: no checkpoint found")
                continue
            print(f"  eval {run_dir.name}")
            try:
                result = evaluate_one(
                    ckpt_path=ckpt,
                    split="pakistan2022",
                    sen1floods11_root=None,
                    polarimetric_mode=None,
                    batch_size=args.batch_size,
                    pakistan2022_root=args.pakistan2022_root,
                )
            except Exception as e:
                print(f"    FAILED: {e}")
                continue
            print(f"    IoU={result['iou']:.4f}  ECE={result.get('ece', 0):.4f}")
            ious.append(result["iou"])
            eces.append(result.get("ece", 0))
            run_dir_eval_out = run_dir / f"pakistan2022_v2_metrics.json"
            run_dir_eval_out.write_text(json.dumps(result, indent=2))

        if not ious:
            continue

        cell_key = f"{bb}__{peft}"
        # Stretch backbones (Hiera-L, ViT-L, ViT-H) live under `stretch`, not `peft`.
        block = "stretch" if bb in ("sam2_hiera_l", "sam_vit_l", "sam_vit_h") else "peft"
        cell = aggregate.setdefault(block, {}).setdefault(cell_key, {})
        splits = cell.setdefault("splits", {})
        splits["pakistan2022"] = {
            "iou": {
                "mean": float(statistics.mean(ious)),
                "std": float(statistics.stdev(ious)) if len(ious) > 1 else 0.0,
                "n": len(ious),
            },
            "ece": {
                "mean": float(statistics.mean(eces)),
                "std": float(statistics.stdev(eces)) if len(eces) > 1 else 0.0,
                "n": len(eces),
            },
        }
        print(f"  --> patched aggregate[{block}][{cell_key}].splits.pakistan2022 "
              f"(IoU mean={splits['pakistan2022']['iou']['mean']:.4f})")

    args.aggregate.write_text(json.dumps(aggregate, indent=2))
    print(f"[rerun] patched aggregate written to {args.aggregate}")
    print(f"[rerun] backup of pre-patch aggregate at {backup}")
    print("[rerun] Now regenerate tables/figures:")
    print("  python -m model.make_figures --results runs/aggregate_results.json --output thesis/figs")
    print("  python -m model.analysis_extras  # for OOD-gap and other extras")


if __name__ == "__main__":
    main()
