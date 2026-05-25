"""Benchmark inference latency and VRAM per (backbone, PEFT) checkpoint.

Picks the seed=42 checkpoint from each (backbone, peft) cell (if it exists),
loads it once, runs `--n-warmup` warmup passes + `--n-trials` timed passes
on synthetic 3x512x512 inputs, records:
    median ms / mean ms / std ms / p95 ms
    peak VRAM (MiB)

Writes thesis/figs/inference_latency.tex and runs/inference_latency.json.

CLI
---
    python -u -m model.benchmark_inference \\
        --runs-dir runs --output-dir thesis/figs --json-out runs/inference_latency.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.train import FloodLightningModule, load_cfg


def find_cfg_for_run(run_dir: Path) -> Path | None:
    cfg = Path(__file__).parent / "configs" / f"{run_dir.name}.yaml"
    return cfg if cfg.exists() else None


def benchmark_one(ckpt_path: Path, cfg_path: Path, n_warmup: int, n_trials: int,
                  device: str = "cuda:0") -> dict:
    cfg = load_cfg(cfg_path)
    module = FloodLightningModule.load_from_checkpoint(
        str(ckpt_path), cfg=cfg, map_location="cpu",
    ).to(device).eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    x = torch.randn(1, 3, 512, 512, device=device)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = module(x)
        torch.cuda.synchronize(device)
        times_ms = []
        for _ in range(n_trials):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _ = module(x)
            torch.cuda.synchronize(device)
            times_ms.append((time.perf_counter() - t0) * 1000.0)
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    del module
    torch.cuda.empty_cache()
    times_ms.sort()
    return {
        "median_ms": statistics.median(times_ms),
        "mean_ms": statistics.mean(times_ms),
        "std_ms": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        "p95_ms": times_ms[int(0.95 * len(times_ms))] if times_ms else float("nan"),
        "peak_vram_mb": peak_mb,
        "n_warmup": n_warmup,
        "n_trials": n_trials,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--output-dir", type=Path, default=Path("report/figs"))
    p.add_argument("--json-out", type=Path, default=Path("runs/inference_latency.json"))
    p.add_argument("--n-warmup", type=int, default=10)
    p.add_argument("--n-trials", type=int, default=100)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42,
                   help="Which seed's ckpt to benchmark per (backbone, peft) cell.")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for run_dir in sorted(args.runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if not run_dir.name.endswith(f"_seed{args.seed}"):
            continue
        ckpt = run_dir / "last.ckpt"
        cfg = find_cfg_for_run(run_dir)
        if not (ckpt.exists() and cfg):
            continue
        print(f"[bench] {run_dir.name}")
        try:
            r = benchmark_one(ckpt, cfg, args.n_warmup, args.n_trials, args.device)
        except Exception as e:
            print(f"  failed: {e}")
            continue
        print(f"  median {r['median_ms']:.1f} ms, p95 {r['p95_ms']:.1f} ms, "
              f"peak {r['peak_vram_mb']:.0f} MiB")
        results[run_dir.name] = r

    args.json_out.write_text(json.dumps(results, indent=2))
    print(f"[bench] wrote {args.json_out}")

    # LaTeX table: backbone | peft | median ms | p95 ms | peak VRAM.
    # Only the principal (backbone, PEFT) cells go in the headline table;
    # extras (decoder-FT, polari, rank, ViT-H tune, linear probe, data eff)
    # are reported elsewhere and would clutter this one with "?" PEFT rows.
    backbones = ("sam2_hiera_bp", "sam2_hiera_l", "sam_vit_b",
                 "sam_vit_l", "sam_vit_h")
    peft_labels = {"lora": "LoRA", "dora": "DoRA",
                   "convlora": "Conv-LoRA", "adaptformer": "AdaptFormer"}
    backbone_labels = {
        "sam2_hiera_bp": "SAM 2 Hiera-B+", "sam2_hiera_l": "SAM 2 Hiera-L",
        "sam_vit_b": "SAM ViT-B", "sam_vit_l": "SAM ViT-L", "sam_vit_h": "SAM ViT-H",
    }
    rows = []
    for name, r in results.items():
        parts = name.split("_seed")[0]
        if name.startswith("extra_") or name.startswith("polari_"):
            continue
        backbone = next((b for b in backbones if parts.startswith(b + "_")), None)
        if backbone is None:
            continue
        peft = parts[len(backbone) + 1:]
        if peft not in peft_labels:
            continue
        rows.append((backbone, peft, r))
    rows.sort(key=lambda x: (x[0], x[1]))

    lines = [r"\begin{tabular}{llccc}",
             r"\toprule",
             r"Backbone & PEFT & Median (ms) & p95 (ms) & Peak VRAM (MiB) \\",
             r"\midrule"]
    for backbone, peft, r in rows:
        lines.append(
            rf"{backbone_labels[backbone]} & {peft_labels[peft]} & "
            rf"{r['median_ms']:.1f} & {r['p95_ms']:.1f} & {r['peak_vram_mb']:.0f} \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (args.output_dir / "inference_latency.tex").write_text("\n".join(lines) + "\n")
    print(f"[bench] wrote {args.output_dir / 'inference_latency.tex'}")


if __name__ == "__main__":
    main()
