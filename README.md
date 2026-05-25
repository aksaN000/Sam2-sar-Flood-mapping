# SAM 2 for Sentinel-1 SAR Flood Inundation Mapping

Parameter-efficient adaptation of the Segment Anything Model 2 (SAM 2) to Sentinel-1 Synthetic Aperture Radar (SAR) flood inundation mapping, evaluated in the Indo-Gangetic Region.

Authors: **Aksan Gony Alif** and **Mahbubur Rahman** (BRAC University).

## What's in this repository

- `model/` — training, evaluation, and analysis code (PyTorch Lightning + HuggingFace Transformers)
- `model/configs/` — YAML configs for every (backbone, PEFT, seed) combination in the sweep
- `future_work/pakistan_2022/` — Pakistan-2022 test-set acquisition pipeline (TU Wien labels + Microsoft Planetary Computer Sentinel-1 RTC). `acquire_v2.py` is the current 10 m native-resolution pipeline used in the paper.
- `future_work/bangladesh_pipeline/` — Bangladesh-2022-Sylhet test-set construction pipeline (shipped as future work, not empirically evaluated in this paper).
- `report/` — LaTeX source and compiled PDFs for both the IEEE conference paper (`paper_ieee.tex`) and the BRAC University thesis (`research_report_detailed.tex`).
- `PAKISTAN2022_RERUN.md` — operational guide for reproducing the v2 Pakistan-2022 evaluation.
- `run_sweep.ps1`, `run_sweep_2gpu.sh`, `status.ps1` — sweep launchers and progress monitors.

## What the paper contributes

1. Systematic empirical comparison of four PEFT methods (LoRA, DoRA, Conv-LoRA, AdaptFormer) on two SAM-family backbones (SAM ViT-B, SAM 2 Hiera-Base-Plus), plus three larger stretch backbones (SAM ViT-L, SAM ViT-H, SAM 2 Hiera-Large) under Conv-LoRA. Three random seeds per cell.
2. Polarimetric prompt-engineering ablation. The headline empirical finding: dropping the cross-polarization (VH) channel and feeding co-polarization (VV) alone restores SAM-family Pakistan-2022 IoU from 0.09–0.14 to 0.56–0.66 (within 0.01–0.12 of the U-Net baseline).
3. Monte Carlo dropout confidence-aware mechanism with both aggregate calibration (Expected Calibration Error) and pointwise selective-prediction analysis. The substantive negative finding: aggregate calibration does not imply useful per-pixel uncertainty on out-of-distribution events.
4. Public Pakistan-2022 SAR flood test set (39 chips at Sentinel-1's native 10 m resolution), built from TU Wien Sentinel-1 flood-extent labels paired with Microsoft Planetary Computer Sentinel-1 RTC imagery via the released acquisition pipeline.

## Getting started

Install Python 3.11 and the project dependencies:

```bash
pip install -r requirements.txt
```

Download the Sen1Floods11 dataset (about 700 MB):

```bash
python -m model.download_sen1floods11
```

Train one cell:

```bash
python -m model.train --config model/configs/sam2_hiera_bp_lora_seed42.yaml \
    --sen1floods11-root /path/to/sen1floods11
```

Build the Pakistan-2022 test set (requires the TU Wien flood-extent labels; see `PAKISTAN2022_RERUN.md`):

```bash
python -m future_work.pakistan_2022.acquire_v2 \
    --masks-dir /path/to/FLOOD-HM-MASKED \
    --out-dir /path/to/pakistan-2022-chips-v2
```

Re-evaluate the trained checkpoints on the v2 Pakistan-2022 chips:

```bash
python -m model.rerun_pakistan2022 \
    --runs-dir runs \
    --pakistan2022-root /path/to/pakistan-2022-chips-v2
```

## Datasets

- **Sen1Floods11**: 446 hand-labelled Sentinel-1 chips, 10 m resolution, distributed via Google Cloud Storage. Used for training, validation, in-distribution test, and the Bolivia held-out country split.
- **Pakistan-2022 (this work)**: 39 chips built at Sentinel-1's native 10 m resolution from TU Wien Sentinel-1-derived flood-extent labels paired with Microsoft Planetary Computer Sentinel-1 RTC imagery. Used as the strict geographic-and-temporal out-of-distribution test.
- **Bangladesh-2022-Sylhet**: construction pipeline shipped under `future_work/bangladesh_pipeline/`; chips not built or evaluated in this paper.

## Hardware

The full sweep (5 backbones × 4 PEFT methods × 3 seeds + polari ablations + U-Net + zero-shot baselines) completes in approximately 3.5 hours on a dual NVIDIA RTX PRO 5000 Blackwell cloud instance for under ten US dollars.

## Citation

If you use this code or the Pakistan-2022 test set, please cite the paper:

```
@inproceedings{alif2026sam2sar,
  author = {Aksan Gony Alif and Mahbubur Rahman},
  title = {Parameter-Efficient Adaptation of SAM 2 for Sentinel-1 SAR Flood Inundation Mapping in the Indo-Gangetic Region},
  year = {2026},
  organization = {BRAC University}
}
```

## License

This codebase is released under the MIT License (see `LICENSE`).

Underlying data sources retain their respective licenses:
- Sentinel-1 imagery: Copernicus open and free data policy.
- TU Wien Pakistan-2022 flood-extent labels: CC-BY 4.0.
- Microsoft Planetary Computer Sentinel-1 RTC: Copernicus open data policy with anonymous Azure Blob SAS-token access.
