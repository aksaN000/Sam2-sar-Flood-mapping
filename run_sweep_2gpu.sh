#!/bin/bash
# Parallel 2-GPU full sweep launcher for Vast.ai 2x RTX 5000 Ada (or any 2-GPU host).
#
# Strategy: each YAML config is pinned to either GPU 0 or GPU 1 via
# CUDA_VISIBLE_DEVICES. Two configs train concurrently; the script waits
# for both to finish before launching the next pair. The HF model cache
# is shared so the second config of a new backbone benefits from the
# first one's download. We pre-warm the cache with a tiny dry-run so the
# first parallel pair does not double-download the same checkpoint.
#
# Usage:
#     conda activate sam2-sar
#     cd ~/sam2-sar-flood
#     bash run_sweep_2gpu.sh
#
# Logs land in logs/<config_name>.log + an aggregate logs/sweep.log with
# timestamps. Checkpoints land in runs/<config_name>/. The script is
# idempotent at config granularity: a config whose summary.json already
# exists is skipped, so you can ^C and restart.

set -u  # error on unset variables; but NOT -e so a single config crash does not abort the sweep
export PYTHONUNBUFFERED=1

mkdir -p logs runs

LOG="logs/sweep.log"
stamp() { echo "[$(date +%Y-%m-%d\ %H:%M:%S)] $*" | tee -a "$LOG"; }

# ---------------- Config queues ----------------

# Main 24-config sweep: 4 PEFT methods x 2 backbones x 3 seeds
mapfile -t MAIN_CONFIGS < <(ls model/configs/sam_vit_b_*_seed*.yaml \
                              model/configs/sam2_hiera_bp_*_seed*.yaml 2>/dev/null | sort)

# Polarimetric ablations (2 new compositions; ratio is already covered)
mapfile -t POLARI_CONFIGS < <(ls model/configs/polari_diff_*.yaml \
                                 model/configs/polari_single_*.yaml 2>/dev/null | sort)

# Stretch sweep: ViT-L, ViT-H, Hiera-Large with Conv-LoRA across 3 seeds
mapfile -t STRETCH_CONFIGS < <(ls model/configs/sam_vit_l_*_seed*.yaml \
                                  model/configs/sam_vit_h_*_seed*.yaml \
                                  model/configs/sam2_hiera_l_*_seed*.yaml 2>/dev/null | sort)

ALL_CONFIGS=("${MAIN_CONFIGS[@]}" "${POLARI_CONFIGS[@]}" "${STRETCH_CONFIGS[@]}")

stamp "BEGIN full 2-GPU sweep on $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ',' | sed 's/,$//')"
stamp "queue size: ${#MAIN_CONFIGS[@]} main + ${#POLARI_CONFIGS[@]} polari + ${#STRETCH_CONFIGS[@]} stretch = ${#ALL_CONFIGS[@]} configs"

# ---------------- Warm the HF cache ----------------
# Touch each unique backbone once on GPU 0 so the parallel sweep does not
# race on first-time HuggingFace Hub downloads.
stamp "warming HF model cache (one model per backbone)"
declare -A BACKBONES_SEEN=()
for cfg in "${ALL_CONFIGS[@]}"; do
    bb=$(python -c "
from omegaconf import OmegaConf
from pathlib import Path
cfg=OmegaConf.merge(OmegaConf.load('model/configs/_base.yaml'),OmegaConf.load('$cfg'))
print(cfg.model.backbone)
")
    if [ -z "${BACKBONES_SEEN[$bb]:-}" ]; then
        BACKBONES_SEEN[$bb]=1
        stamp "  pre-fetching $bb"
        CUDA_VISIBLE_DEVICES=0 python -u -c "
import sys; sys.path.insert(0, '.')
from model.train import FloodLightningModule, load_cfg
from pathlib import Path
cfg = load_cfg(Path('$cfg'))
mod = FloodLightningModule(cfg)
print('cached', cfg.model.backbone)
" 2>&1 | tail -3 | tee -a "$LOG"
    fi
done
stamp "HF cache warm; starting parallel sweep"

# ---------------- Parallel sweep ----------------

train_one() {
    local gpu="$1"
    local cfg="$2"
    local name
    name=$(basename "$cfg" .yaml)
    if [ -f "runs/$name/summary.json" ]; then
        stamp "  SKIP $name (summary.json already exists)"
        return 0
    fi
    stamp "  GPU$gpu: starting $name"
    CUDA_VISIBLE_DEVICES="$gpu" python -u -m model.train \
        --config "$cfg" \
        --output-dir ./runs 2>&1 | tee "logs/${name}.log"
    if [ -f "runs/$name/summary.json" ]; then
        stamp "  GPU$gpu: done $name"
    else
        stamp "  GPU$gpu: WARNING $name produced no summary.json"
    fi
}

# Process configs in pairs: GPU 0 and GPU 1 in parallel
n=${#ALL_CONFIGS[@]}
for (( i=0; i<n; i+=2 )); do
    cfg0="${ALL_CONFIGS[i]}"
    cfg1=""
    if (( i+1 < n )); then
        cfg1="${ALL_CONFIGS[i+1]}"
    fi

    if [ -n "$cfg1" ]; then
        train_one 0 "$cfg0" &
        PID0=$!
        train_one 1 "$cfg1" &
        PID1=$!
        wait $PID0 $PID1
    else
        train_one 0 "$cfg0"
    fi
done

# ---------------- U-Net baseline (3 seeds) ----------------
stamp "Phase B: U-Net baseline, 3 seeds"
for seed in 42 123 20025; do
    name="unet_seed$seed"
    if [ -f "runs/$name/summary.json" ]; then
        stamp "  SKIP $name (already done)"
        continue
    fi
    stamp "  GPU0: training $name"
    CUDA_VISIBLE_DEVICES=0 python -u -m model.train_unet \
        --seed "$seed" \
        --output-dir ./runs 2>&1 | tee "logs/${name}.log"
done

# ---------------- Zero-shot baselines ----------------
stamp "Phase C: zero-shot baselines"
CUDA_VISIBLE_DEVICES=0 python -u -m model.eval_zeroshot \
    --backbone sam-vit-b \
    --split test --split bolivia --split pakistan --split pakistan2022 \
    --pakistan2022-root ./data/pakistan-2022-chips \
    2>&1 | tee logs/zeroshot_sam_vit_b.log
CUDA_VISIBLE_DEVICES=0 python -u -m model.eval_zeroshot \
    --backbone sam2-hiera-bp \
    --split test --split bolivia --split pakistan --split pakistan2022 \
    --pakistan2022-root ./data/pakistan-2022-chips \
    2>&1 | tee logs/zeroshot_sam2_hiera_bp.log

# ---------------- Aggregate ----------------
stamp "Phase D: aggregate"
python -u -m model.aggregate_results \
    --runs-dir ./runs \
    --output ./runs/aggregate_results.json \
    --splits test bolivia pakistan pakistan2022 \
    --pakistan2022-root ./data/pakistan-2022-chips \
    2>&1 | tee logs/aggregate.log

# ---------------- Figures ----------------
stamp "Phase E: figures"
python -u -m model.make_figures \
    --results runs/aggregate_results.json \
    --output thesis/figs \
    2>&1 | tee logs/figures.log

stamp "DONE  -- aggregate_results.json is at runs/aggregate_results.json"
stamp "       figures are under thesis/figs/"
stamp "       commit + push the JSON + figs back to main and stop the Vast.ai instance"
