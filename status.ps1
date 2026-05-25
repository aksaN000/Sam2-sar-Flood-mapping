# Quick status snapshot for the running sweep.
# Run it any time:
#     .\status.ps1
#
# Shows: GPU utilization + VRAM, sweep.log tail, list of completed runs
# with their best val IoU, and which config is currently training.

$ErrorActionPreference = "Continue"
$projectRoot = $PSScriptRoot
Set-Location $projectRoot

Write-Host ""
Write-Host "===== GPU =====" -ForegroundColor Cyan
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv

Write-Host ""
Write-Host "===== sweep.log (last 15 lines) =====" -ForegroundColor Cyan
if (Test-Path logs\sweep.log) {
    Get-Content logs\sweep.log -Tail 15
} else {
    Write-Host "(no sweep.log yet)"
}

Write-Host ""
Write-Host "===== completed runs =====" -ForegroundColor Cyan
$completed = Get-ChildItem runs -Directory -ErrorAction SilentlyContinue |
    Where-Object { Test-Path (Join-Path $_.FullName 'summary.json') }
if ($completed) {
    foreach ($run in $completed) {
        $summaryPath = Join-Path $run.FullName 'summary.json'
        try {
            $summary = Get-Content $summaryPath -Raw | ConvertFrom-Json
            $iou = if ($summary.best_val_iou) { "{0:F4}" -f $summary.best_val_iou } else { "n/a" }
            $params = if ($summary.trainable_params) {
                "{0:F2}M" -f ($summary.trainable_params / 1e6)
            } else { "n/a" }
            "{0,-45} val_iou={1,-8} trainable={2}" -f $run.Name, $iou, $params | Write-Host
        } catch {
            "{0,-45} (summary.json unreadable)" -f $run.Name | Write-Host
        }
    }
    Write-Host ""
    Write-Host ("  total completed: {0}" -f $completed.Count)
} else {
    Write-Host "(no completed runs yet)"
}

Write-Host ""
Write-Host "===== current activity =====" -ForegroundColor Cyan
# Detect which config is currently training by looking at the most recently
# touched run directory that does NOT have summary.json.
$inProgress = Get-ChildItem runs -Directory -ErrorAction SilentlyContinue |
    Where-Object { -not (Test-Path (Join-Path $_.FullName 'summary.json')) } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if ($inProgress) {
    $ckpts = Get-ChildItem (Join-Path $inProgress.FullName '*.ckpt') -ErrorAction SilentlyContinue
    $latestCkpt = $ckpts | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    Write-Host ("  config:        {0}" -f $inProgress.Name)
    if ($latestCkpt) {
        $age = (Get-Date) - $latestCkpt.LastWriteTime
        Write-Host ("  latest ckpt:   {0} ({1:N1} min ago)" -f $latestCkpt.Name, $age.TotalMinutes)
    } else {
        Write-Host "  latest ckpt:   (none yet, just started)"
    }
} else {
    Write-Host "(no in-progress run detected)"
}

Write-Host ""
