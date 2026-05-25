# Configurations

All training and ablation configurations are YAML files in this directory. Every config inherits defaults from `_base.yaml` and overrides only what differs from the base. The total config count is 76 across nine groups.

## Inheritance pattern

Each config file declares `defaults: [- _base]` at the top and then overrides specific keys. The `load_cfg` function in `model/train.py` merges the override file with `_base.yaml` via `OmegaConf.merge(base, override)` so the resolved config always has every key the trainer needs.

Example minimal override:

```yaml
defaults:
  - _base

model:
  backbone: sam2-hiera-bp
  peft: lora

train:
  seed: 42

dataset:
  polarimetric_mode: ratio
```

The resolved config is a flat dictionary containing the full `_base` keys with the three overrides applied.

## Naming convention

Configs are named by a hyphen-free `<backbone>_<peft>_seed<N>.yaml` pattern. Extra ablation configs prepend `extra_` and a category prefix. The categories are:

| Prefix | Group | Example | What it ablates |
|---|---|---|---|
| `<backbone>_<peft>_seed<N>` | peft (main sweep) | `sam2_hiera_bp_lora_seed42` | PEFT method on the principal backbones |
| `polari_<mode>_sam2_convlora_seed42` | polari (mode ablation) | `polari_diff_sam2_convlora_seed42` | Polarimetric pseudo-RGB composition |
| `sam_vit_l_convlora_seed<N>`, `sam_vit_h_convlora_seed<N>`, `sam2_hiera_l_convlora_seed<N>` | stretch | `sam_vit_h_convlora_seed42` | Backbone scale |
| `unet_seed<N>` | unet baseline | `unet_seed42` | U-Net ResNet-34 with same data pipeline |
| `confidence_mcdropout_sam2_convlora_seed42` | confidence eval | (same name) | MC dropout configuration parameters |
| `extra_linearprobe_<backbone>_seed<N>` | linearprobe | `extra_linearprobe_sam_vit_b_seed42` | Frozen-encoder, decoder-only LoRA |
| `extra_polari_<mode>_sam2_<peft>_seed42` | polari_pi (interaction) | `extra_polari_diff_sam2_dora_seed42` | Polarimetric mode crossed with non-ConvLoRA PEFT |
| `extra_decoderft_sam2_<peft>_seed42` | decoder full-FT | `extra_decoderft_sam2_lora_seed42` | Full fine-tune of mask decoder (4 PEFT methods, seed 42 only) |
| `extra_dataeff_<pct>pct_sam2_adaptformer_seed42` | data efficiency | `extra_dataeff_25pct_sam2_adaptformer_seed42` | Training-data subsample fraction |
| `extra_rank<r>_sam2_lora_seed<N>` | LoRA rank | `extra_rank32_sam2_lora_seed42` | LoRA rank in {4, 16, 32} (rank 8 is in main sweep) |
| `extra_vith_tune_seed<N>` | ViT-H HP retune | `extra_vith_tune_seed42` | Larger rank, lower LR, for ViT-H |
| `extra_long_sam2_adaptformer_seed<N>` | long training | `extra_long_sam2_adaptformer_seed42` | 30 epochs of the headline cell (explicitly skipped in the sweep) |

## Count per group

| Group | Count | Configurations |
|---|---|---|
| peft (main) | 24 | 4 PEFT methods × 2 backbones × 3 seeds (42, 123, 20025) |
| polari | 3 | `ratio`, `diff`, `single` modes (seed 42 only; ratio is shared with main sweep) |
| stretch | 9 | 3 stretch backbones × 1 PEFT × 3 seeds |
| unet | 3 | U-Net ResNet-34, seeds 42 / 123 / 20025 |
| confidence | 1 | mc dropout settings only (training is shared with main sweep) |
| linearprobe | 6 | 2 backbones × 3 seeds, frozen encoder |
| polari_pi | 6 | 2 non-ConvLoRA polari modes × 3 non-ConvLoRA PEFT methods (seed 42 only) |
| decoderft | 4 | 4 PEFT methods × seed 42 |
| dataeff | 4 | 4 data fractions × seed 42 |
| rank | 9 | 3 LoRA ranks × 3 seeds |
| vith_tune | 3 | ViT-H with rank=16 + lower LR × 3 seeds |
| long | 3 | 30-epoch of the headline cell × 3 seeds (NOT trained in this sweep) |
| **Total** | **76** | including `_base.yaml` |

## Hyperparameter defaults (from `_base.yaml`)

| Section | Key | Default | Notes |
|---|---|---|---|
| `dataset` | `sen1floods11_root` | `./data/sen1floods11` | Location of the downloaded chips |
| `dataset` | `polarimetric_mode` | `ratio` | One of `ratio`, `diff`, `single` |
| `model` | `backbone` | `sam2-hiera-bp` | One of `sam-vit-b/l/h`, `sam2-hiera-bp/l` |
| `model` | `peft` | `lora` | One of `lora`, `dora`, `conv-lora`, `adaptformer`, `none` (linear-probe) |
| `model` | `rank` | `8` | LoRA / DoRA / ConvLoRA rank |
| `model` | `adaptformer_dim` | `64` | AdaptFormer bottleneck dimension |
| `model` | `mask_decoder_strategy` | `lora-r4` | LoRA rank 4 on the mask decoder (uniform across PEFT methods) |
| `model` | `use_memory` | `false` | SAM 2 streaming memory module: disabled for single-frame Sen1Floods11 |
| `train` | `seed` | `42` | Per-config override picks 42, 123, or 20025 |
| `train` | `finetune_epochs` | `10` | Bumped to 30 for the explicitly-skipped `extra_long` configs |
| `train` | `batch_size` | `1` | With `grad_accum=4` -> effective batch 4 |
| `train` | `grad_accum` | `4` | Bumped to 8 for the OOM-prone ViT-H |
| `train` | `precision` | `16-mixed` | fp16-mixed |
| `train` | `num_workers` | `2` | |
| `train` | `optimizer.name` | `adamw` | |
| `train` | `optimizer.lr_adapter` | `1.0e-4` | LR for the PEFT and decoder-LoRA parameters |
| `train` | `optimizer.lr_unfrozen` | `1.0e-5` | LR for the unfrozen base modules (used in decoder full-FT) |
| `train` | `optimizer.weight_decay` | `0.01` | |
| `train` | `optimizer.betas` | `[0.9, 0.999]` | |
| `train` | `scheduler.warmup_steps` | `200` | |
| `train` | `scheduler.schedule` | `cosine` | Cosine decay to zero by epoch 10 |
| `train` | `loss.dice_weight` | `1.0` | Soft dice loss component |
| `train` | `loss.bce_weight` | `1.0` | Binary cross-entropy component |
| `eval` | `test_splits` | `[test, bolivia, pakistan]` | Default eval splits (pakistan2022 added by aggregate.py) |
| `eval` | `ece_bins` | `15` | Number of bins for ECE computation |
| `confidence` | `mode` | `none` | One of `none`, `mc_dropout`, `deep_ensemble` |
| `confidence` | `mc_passes` | `20` | MC dropout forward passes |
| `confidence` | `ensemble_seeds` | `[42, 123, 20025, 7, 2024]` | Seeds for deep ensemble (extra seeds 7 and 2024 not trained in this sweep) |
| `logging` | `wandb_mode` | `offline` | WandB stays offline for reproducibility |
| `logging` | `wandb_project` | `sam2-sar-flood` | |
| `logging` | `output_dir` | `runs` | Per-config dir is `runs/<config_name>/` |

## How to invoke

The trainer reads a single config file:

```bash
python -m model.train --config model/configs/sam2_hiera_bp_lora_seed42.yaml
```

The trainer creates `runs/<config_name>/` and writes `last.ckpt`, `best-{epoch}-{val_iou}.ckpt`, and `summary.json`. The aggregate / analysis scripts walk this directory tree to discover configurations.

To override a single key without editing the YAML, use the `--override` flag:

```bash
python -m model.train --config model/configs/sam2_hiera_bp_lora_seed42.yaml \
    --override train.finetune_epochs=20
```

## Notes about specific configs

- **Stub `unet_seed{42,123,20025}.yaml`** were added in the second-instance recovery (see `RETRY_LOG.md`). They exist only so `model/eval.py:find_cfg_for_ckpt` can resolve a config path for U-Net checkpoints; the actual model is built by `model/train_unet.py:UNetModule` directly from `_base.yaml` and ignores the `model.backbone` / `model.peft` keys.
- **`polari_ratio_sam2_convlora_seed42`** is a no-op alias of `sam2_hiera_bp_convlora_seed42` (the main sweep already runs the `ratio` polarimetric mode by default). The aggregate script treats the main-sweep `sam2_hiera_bp_convlora_seed42` as the `ratio` cell in the polarimetric ablation table.
- **`polari_diff_sam2_convlora_seed42`** uses the `diff` polarimetric mode which is mathematically identical to `ratio` under dB scaling. See the Limitations section of the thesis; this ablation cell is included for completeness but does not test a distinct input encoding.
- **`extra_vith_tune_seed{42,123,20025}.yaml`** uses LoRA rank 16 and a 10x lower learning rate. The result (~0.16 val IoU) is substantially worse than the default ViT-H sweep (~0.47 val IoU); the tune attempt is honest disclosure rather than the headline ViT-H number.
