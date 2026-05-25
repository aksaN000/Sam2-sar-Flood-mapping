# Automated overnight sweep: PEFT main + polarimetric ablations + U-Net baseline +
# zero-shot baselines + aggregation + figures.
#
# Run this from the project root with the sam2-sar conda env active:
#     conda activate sam2-sar
#     cd "C:\Users\aksan\Downloads\files (1)\sam2-sar-flood"
#     .\run_sweep.ps1
#
# Everything logs to logs\<step>.log plus an aggregate logs\sweep.log with timestamps.
# Each step pauses for any in-flight Python to exit, so the GPU is never double-used.

# Keep ErrorActionPreference at Continue so harmless Python stderr output
# (warnings written via the logging module, tqdm progress, etc.) does not
# abort the sweep. We check $LASTEXITCODE manually after each python call.
$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"   # so progress prints flush through Tee-Object
New-Item -ItemType Directory -Force logs, runs | Out-Null

function Invoke-Python([string]$argString, [string]$logFile) {
    $cmd = "python -u $argString 2>&1 | Tee-Object -FilePath '$logFile'"
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Stamp "  WARNING: python exited with code $LASTEXITCODE; continuing"
    }
}

function Write-Stamp([string]$msg) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$stamp] $msg" | Tee-Object -FilePath logs\sweep.log -Append
}

Write-Stamp "BEGIN automated sweep"

# ---------------- Phase 1: main PEFT sweep ----------------
# 24 configs total. Skip the one we already trained in the smoke test.
$mainConfigs = Get-ChildItem model\configs\sam*_seed*.yaml |
  Where-Object { $_.BaseName -ne "sam2_hiera_bp_lora_seed42" }
Write-Stamp ("Phase 1: " + $mainConfigs.Count + " main PEFT configs")

foreach ($cfg in $mainConfigs) {
    $name = $cfg.BaseName
    Write-Stamp "  starting $name"
    Invoke-Python "-m model.train --config `"$($cfg.FullName)`"" "logs\$name.log"
    if (-not (Test-Path "runs\$name\summary.json")) {
        Write-Stamp "  WARNING: $name did not produce summary.json"
    } else {
        Write-Stamp "  done $name"
    }
}

# ---------------- Phase 2: polarimetric ablation ----------------
# 'ratio' is already covered by sam2_hiera_bp_convlora_seed42 (the same model with
# default ratio composition), so only 'diff' and 'single' need a separate run.
$polariConfigs = @(
    "model\configs\polari_diff_sam2_convlora_seed42.yaml",
    "model\configs\polari_single_sam2_convlora_seed42.yaml"
)
Write-Stamp ("Phase 2: " + $polariConfigs.Count + " polarimetric ablation configs")

foreach ($cfgPath in $polariConfigs) {
    $name = [IO.Path]::GetFileNameWithoutExtension($cfgPath)
    Write-Stamp "  starting $name"
    Invoke-Python "-m model.train --config `"$cfgPath`"" "logs\$name.log"
    Write-Stamp "  done $name"
}

# ---------------- Phase 3: U-Net baseline (3 seeds) ----------------
Write-Stamp "Phase 3: U-Net baseline, 3 seeds"
foreach ($seed in 42, 123, 20025) {
    Write-Stamp "  starting unet_seed$seed"
    Invoke-Python "-m model.train_unet --seed $seed" "logs\unet_seed$seed.log"
    Write-Stamp "  done unet_seed$seed"
}

# ---------------- Phase 4: zero-shot baselines ----------------
Write-Stamp "Phase 4: zero-shot SAM and SAM 2"
Invoke-Python "-m model.eval_zeroshot --backbone sam-vit-b --split test --split bolivia --split pakistan" "logs\zeroshot_sam_vit_b.log"
Invoke-Python "-m model.eval_zeroshot --backbone sam2-hiera-bp --split test --split bolivia --split pakistan" "logs\zeroshot_sam2_hiera_bp.log"

# ---------------- Phase 5: aggregate results ----------------
Write-Stamp "Phase 5: aggregate checkpoints across seeds"
Invoke-Python "-m model.aggregate_results --runs-dir runs --output runs\aggregate_results.json" "logs\aggregate.log"

# ---------------- Phase 6: figures + LaTeX table ----------------
Write-Stamp "Phase 6: generate figures and LaTeX table"
Invoke-Python "-m model.make_figures --results runs\aggregate_results.json --output thesis\figs" "logs\figures.log"

Write-Stamp "END automated sweep -- check runs\aggregate_results.json and thesis\figs\"
