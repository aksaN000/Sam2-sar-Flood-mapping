# Models, training, and evaluation

This directory holds the implementation code and configurations for adapting SAM and SAM 2 to Sentinel-1 SAR flood inundation mapping. Every script here is invoked via Python module syntax from the project root (`python -m model.<name>`).

## Architecture overview

The thesis evaluates two SAM-family backbones in the principal sweep and three additional stretch backbones, all wrapped by adapter classes that inject parameter-efficient fine-tuning modules into the frozen image encoder. The two principal backbones are SAM ViT-B (the original Segment Anything Vision Transformer Base) and SAM 2 Hiera-Base-Plus (the SAM 2 hierarchical Vision Transformer with a streaming memory module, which is disabled by setting `use_memory: false` in `_base.yaml`). The three stretch backbones add larger SAM variants (ViT-L, ViT-H) and the larger SAM 2 (Hiera-Large) to ablate the parameter scale.

All backbones consume a three-channel pseudo-RGB input constructed from the dual-polarization Sentinel-1 signal as documented in `data/README.md`. The output is a single-channel logit map at the original 512x512 resolution, converted to a flood-probability mask via sigmoid and thresholded at 0.5 for the principal IoU metric.

## Code layout

```
model/
├── __init__.py
├── README.md (this file)
├── configs/                                 76 YAML configuration files (see configs/README.md)
├── datasets/
│   ├── __init__.py
│   ├── polarimetric.py                       pseudo-RGB compositions + percentile-clip + z-score normalization
│   └── sen1floods11.py                       PyTorch Dataset, Sen1Floods11 + Pakistan-2022, six named splits
├── models/
│   ├── __init__.py
│   ├── peft_methods.py                       LoRA, DoRA, ConvLoRA, AdaptFormer wrappers around nn.Linear
│   ├── sam_adapter.py                        SAM ViT-B/L/H + selected PEFT method
│   ├── sam2_adapter.py                       SAM 2 Hiera-Base-Plus/Large + selected PEFT method
│   └── confidence.py                         MCDropoutPredictor, DeepEnsemblePredictor, expected_calibration_error
├── train.py                                  Lightning training entry point (FloodLightningModule for SAM-family)
├── train_unet.py                             U-Net ResNet-34 baseline trainer (UNetModule, distinct module class)
├── eval.py                                   inference + metric computation, with U-Net dispatch
├── eval_zeroshot.py                          zero-shot SAM / SAM 2 baseline (no training)
├── eval_confidence.py                        Monte Carlo dropout reliability binning
├── eval_ensemble.py                          deep ensemble + TTA + temperature scaling
├── aggregate_results.py                      walks runs/, evaluates each config on each split, writes JSON
├── benchmark_inference.py                    per-checkpoint inference latency benchmark
├── analysis_selective.py                     MC dropout selective-prediction curves
├── analysis_failures.py                      worst-IoU chip galleries on OOD splits
├── analysis_extras.py                        13 LaTeX tables: stats / efficiency / convergence / OOD gap / etc.
├── make_figures.py                           five principal figure PDFs from aggregate_results.json
└── download_sen1floods11.py                  public-bucket downloader for Sen1Floods11
```

## Parameter-efficient fine-tuning methods

Four PEFT methods are implemented in `model/models/peft_methods.py`. All four are applied uniformly to the query and value projection matrices of every transformer block in the image encoder. The mask decoder uses a separate LoRA rank-4 head regardless of which encoder PEFT is selected, controlled by `mask_decoder_strategy: lora-r4` in `_base.yaml`. Each method's trainable parameter percentage (relative to the total backbone parameter count) is reported in `thesis/figs/efficiency_table.tex`.

### LoRA (Low-Rank Adaptation)

Hu et al., ICLR 2022. Approximates the weight update during fine-tuning as a low-rank decomposition `dW = B @ A` where `B` is `d x r` and `A` is `r x d` with `r << d`. The pretrained weight `W` is frozen; only `A` and `B` are trained. The default rank is 8, ablated over 4, 16, and 32 in `extra_rank*_*.yaml` configs.

### DoRA (Weight-Decomposed Low-Rank Adaptation)

Liu et al., ICML 2024 Oral. Refines LoRA by decomposing the pretrained weight into a magnitude vector and a directional matrix, and applies LoRA-style updates only to the directional component. The magnitude vector is initialized to the column-wise L2 norm of `W` and trained directly (no low-rank constraint on it). Closes a substantial portion of the gap between LoRA and full fine-tuning at essentially the same parameter cost.

### Conv-LoRA (Convolutional LoRA)

Zhong et al., ICLR 2024. Augments the LoRA decomposition with an ultra-lightweight depthwise convolution on the low-rank intermediate, injecting locality-aware inductive bias into the otherwise plain Vision Transformer encoder. The convolution kernel is depthwise 3x3 with no bias, applied between `A` and `B` of the LoRA factorization.

### AdaptFormer

Chen et al., NeurIPS 2022. Inserts a parallel bottleneck feed-forward module into each transformer block. The bottleneck is `down: hidden_dim -> bottleneck`, `activation: GELU`, `up: bottleneck -> hidden_dim`, followed by a learnable scalar scale. The default bottleneck dimension is 64. Initialization is critical: `up.weight = 0` so the side path output is initially zero (identity-preserving), but `scale = 1` so the gradient with respect to `up.weight` is nonzero. Both LoRA-style identity-init and AdaptFormer-style zero-init require careful attention; see `RETRY_LOG.md` for the disclosed init bug fixed during the sweep.

## Confidence-aware mechanism

Monte Carlo dropout is implemented in `model/models/confidence.py:MCDropoutPredictor`. At inference, the model is set to `.eval()` mode but every `nn.Dropout` submodule is forced into training mode so its randomness stays active. `n_passes` forward passes are run on each batch (default 20) and the per-pixel mean and standard deviation of the sigmoid probabilities are returned. The mean is the final flood-probability map; the standard deviation is the per-pixel epistemic uncertainty estimate. Expected calibration error is computed from the mean probabilities via `expected_calibration_error` with 15 equal-width bins (Guo et al. ICML 2017).

A deep ensemble alternative is also available via `DeepEnsemblePredictor`, which averages over K independently trained adapter checkpoints (different random seeds). Used in `eval_ensemble.py` for the deep ensemble + TTA + temperature scaling results.

## Training protocol

All configurations train for 10 epochs on 252 Sen1Floods11 hand-labeled training chips with batch size 1 and gradient accumulation 4 (effective batch 4). Precision is fp16-mixed. Optimizer is AdamW with `lr_adapter=1e-4` for the PEFT and decoder-LoRA parameters and `lr_unfrozen=1e-5` for any module that is unfrozen (which is none in the principal sweep but is used in the decoder full-FT ablation). Learning rate is warmed up linearly for 200 steps then decayed cosine to zero by epoch 10. Loss is the unweighted sum of soft dice loss and binary cross-entropy, both computed only over non-ignore pixels (label != 255). All hyperparameters are documented per-cell in `_base.yaml` and per-config overrides in each `*.yaml` file.

## Evaluation

The aggregate_results.py script walks `runs/`, applies a regex over directory names to group configs into the seven main categories (peft, stretch, polari, linearprobe, polari_pi, unet, zeroshot), then evaluates each checkpoint on the four canonical splits (test, bolivia, pakistan, pakistan2022). For each (group, cell, split, seed) it computes IoU (the principal metric), F1, precision, recall, and expected calibration error. Per-split metrics are aggregated across seeds via mean and standard deviation. The JSON output is `runs/aggregate_results.json`.

Each downstream analysis script writes its own JSON: confidence reliability (`runs/confidence_test.json`), inference latency benchmark (`runs/inference_latency.json`), selective-prediction curves (`runs/selective_prediction.json`), worst-IoU failure chips (`runs/failure_cases.json`), deep ensemble + TTA + temperature (`runs/ensemble_results.json`). `make_figures.py` reads `runs/aggregate_results.json` and writes five principal figure PDFs plus the headline results table. `analysis_extras.py` reads `runs/aggregate_results.json` plus per-config `summary.json` files and writes 13 additional LaTeX tables to `thesis/figs/`.

## How to train a single config

```bash
python -m model.train --config model/configs/sam2_hiera_bp_lora_seed42.yaml
```

Optional environment variables:

- `PYTHONUNBUFFERED=1` ensures stdout is unbuffered, which is essential when piping training output through `tee` or PowerShell `Tee-Object`.
- `LD_PRELOAD=/venv/main/lib/libjpeg.so.8` on Linux/Vast.ai instances avoids a libjpeg/libtiff symbol collision when rasterio is imported after torch.
- `HF_HOME=/path/to/cache` redirects the HuggingFace download cache; required if `/workspace/.hf_home` has stale-file-handle artifacts from a prior crashed run.
- `TRAINER_ENABLE_GRAD_CKPT=1` enables HuggingFace gradient checkpointing on the backbone, used as an OOM recovery for ViT-H training.

## How to run inference

```bash
python -m model.aggregate_results \
    --runs-dir runs --output runs/aggregate_results.json \
    --splits test bolivia pakistan pakistan2022 \
    --sen1floods11-root ./data/sen1floods11 \
    --pakistan2022-root ./data/pakistan-2022-chips
```

The script discovers checkpoints via the per-config directory naming convention (`sam_vit_b__lora`, `sam2_hiera_bp__adaptformer`, etc.), prefers `best-*.ckpt` over `last.ckpt` (the best-val-IoU snapshot), and dispatches to either `FloodLightningModule` (for SAM-family) or `UNetModule` (for `unet_seed*` runs) at load time. See `RETRY_LOG.md` for the U-Net dispatch bug that was fixed in the second-instance recovery.

## Configuration files

See `model/configs/README.md` for the configuration naming convention, the hyperparameter table per group, and how to invoke a specific config.
