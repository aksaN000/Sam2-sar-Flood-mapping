# Pakistan-2022 targeted rerun

## What this fixes

The original Pakistan-2022 chips were built at **20 m** resolution to match the
TU Wien mask grid, while the Sen1Floods11 training distribution is at native
**10 m**. That 2x scale mismatch was confounding the Pakistan-2022 IoU
numbers with a resolution artifact unrelated to genuine OOD shift.

The fix: keep Sentinel-1 imagery at its native 10 m UTM grid and reproject
the TU Wien labels up to 10 m via nearest-neighbor. Chips become 512x512 at
10 m (5.12 x 5.12 km each), matching Sen1Floods11 exactly.

The fix is **acquisition-side only**. No retraining is required because
the existing checkpoints were trained on 10 m Sen1Floods11 chips; they
were just being tested on the wrong-scale Pakistan-2022 chips.

## Files added

- `data_pipelines/pakistan_2022/acquire_v2.py` — fixed acquisition pipeline.
- `model/rerun_pakistan2022.py` — re-evaluates existing checkpoints on the
  new v2 chips and patches `runs/aggregate_results.json` in place (the
  pre-patch aggregate is backed up to `runs/aggregate_results.before_v2.json`).

## End-to-end workflow

### Step 1. Make TU Wien Pakistan-2022 masks available locally
Either re-download from <https://researchdata.tuwien.at/records/zvvmh-nan78>
or copy the existing `D:/datasets/pakistan-2022/` directory (911 MB, 164
mask tiles) to whichever machine runs the rerun.

### Step 2. Rebuild the chip set at 10 m
```bash
conda activate sam2-sar
python -m data_pipelines.pakistan_2022.acquire_v2 \
    --masks-dir D:/datasets/pakistan-2022/FLOOD-HM-MASKED \
    --out-dir   D:/datasets/pakistan-2022-chips-v2
```
This needs internet (Planetary Computer STAC + signed-URL S1 downloads).
No GPU required. Expect ~60-90 chips (slightly more than v1's 58 because
the chip size in physical area is now smaller).

### Step 3. Download checkpoints from GDrive
Pull `runs/` from Google Drive onto the local machine. Required dirs:
```
runs/
  sam_vit_b_lora_seed{42,123,20025}/best-*.ckpt
  sam_vit_b_dora_seed{42,123,20025}/best-*.ckpt
  sam_vit_b_convlora_seed{42,123,20025}/best-*.ckpt
  sam_vit_b_adaptformer_seed{42,123,20025}/best-*.ckpt
  sam2_hiera_bp_lora_seed{42,123,20025}/best-*.ckpt
  sam2_hiera_bp_dora_seed{42,123,20025}/best-*.ckpt
  sam2_hiera_bp_convlora_seed{42,123,20025}/best-*.ckpt
  sam2_hiera_bp_adaptformer_seed{42,123,20025}/best-*.ckpt
  sam2_hiera_l_convlora_seed{42,123,20025}/best-*.ckpt
  sam_vit_l_convlora_seed{42,123,20025}/best-*.ckpt
  sam_vit_h_convlora_seed{42,123,20025}/best-*.ckpt
  unet_seed{42,123,20025}/best-*.ckpt
```
The polari and ablation runs are not needed here (their Pakistan-2022
numbers are not reported in the main table; if you want them refreshed
too, add their dirs and they will be picked up automatically by the
seed-grouping logic — only main-sweep dirs match the parse pattern).

### Step 4. Run the targeted re-evaluation
```bash
python -m model.rerun_pakistan2022 \
    --runs-dir runs \
    --pakistan2022-root D:/datasets/pakistan-2022-chips-v2 \
    --aggregate runs/aggregate_results.json
```
On the local RTX 3060 (12 GB) this should take ~10 to 20 minutes (~60
chips x ~30 cells x sub-second-per-chip inference). The script:
1. Backs up `runs/aggregate_results.json` to
   `runs/aggregate_results.before_v2.json`.
2. Walks the matching `runs/<backbone>_<peft>_seed<*>/` dirs.
3. Picks `best-*.ckpt` from each.
4. Calls `model.eval.evaluate_one(..., split="pakistan2022", ...)` using the
   v2 chip root.
5. Writes per-run results to `runs/<run>/pakistan2022_v2_metrics.json`.
6. Patches the `splits.pakistan2022` block of each `peft.<backbone>__<peft>`
   cell in the aggregate.

### Step 5. Regenerate tables and figures
```bash
python -m model.make_figures \
    --results runs/aggregate_results.json \
    --output report/figs
python -m model.analysis_extras \
    --runs-dir runs \
    --results runs/aggregate_results.json \
    --output-dir report/figs
```
This refreshes `results_table.tex`, `ood_gap.tex`, and any other figure
that reads from `aggregate_results.json`.

### Step 6. Recompile the paper
```bash
cd report
pdflatex paper_ieee.tex && pdflatex paper_ieee.tex
pdflatex research_report_detailed.tex && pdflatex research_report_detailed.tex
```

## After the rerun

The expected outcome (depending on whether 10 m resampling was a major
driver):

- **Absolute Pakistan-2022 IoU likely rises** for all cells if the 20 m
  artifact was material; the relative ranking of methods may or may not
  hold.
- **The polari single-pol > dual-pol reversal** on Pakistan-2022 may
  disappear, attenuate, or persist. If it persists, it is a real OOD
  finding rather than a reprojection artifact on the VH channel.
- **The Bolivia and Sen1Floods11-test numbers do not change** (they were
  not affected by the Pakistan-2022 pipeline).

After the rerun, the paper's discussion paragraphs on Pakistan-2022 (Polari
ablation, OOD Gap, Discussion) may need a one-line update if the
qualitative picture changes. The Discussion's headline finding (LoRA
family generalizes better than AdaptFormer on Bolivia) is independent of
this fix and will not change.
