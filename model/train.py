"""Training entry point for the SAM/SAM 2 SAR flood adapters.

Reads a YAML config from model/configs/, fine-tunes the chosen backbone+PEFT
combination on Sen1Floods11 hand-labeled chips, writes one checkpoint per
run plus a JSON record of train/val/test metrics. Uses PyTorch Lightning
for the training loop and torchmetrics for IoU / F1 / precision / recall.

CLI
---
    python -m model.train --config model/configs/sam2_hiera_bp_lora_seed42.yaml
    python -m model.train --config <yaml> --epochs 3   # quick override
    python -m model.train --config <yaml> --sen1floods11-root D:/datasets/sen1floods11
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryJaccardIndex,
    BinaryPrecision,
    BinaryRecall,
)

# Allow `python -m model.train` even when run from a sister directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.datasets.sen1floods11 import Sen1Floods11Dataset
from model.models.sam_adapter import SAMAdapter
from model.models.sam2_adapter import SAM2Adapter


# ---------------- losses ----------------

def dice_loss(logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor,
              eps: float = 1.0) -> torch.Tensor:
    """Soft Dice on the flood class, ignoring `valid==0` pixels."""
    probs = torch.sigmoid(logits)
    probs = probs * valid
    t = targets.float() * valid
    inter = (probs * t).sum(dim=(-2, -1))
    denom = probs.sum(dim=(-2, -1)) + t.sum(dim=(-2, -1))
    return (1 - (2 * inter + eps) / (denom + eps)).mean()


def bce_loss(logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy with logits, masked by `valid`."""
    raw = F.binary_cross_entropy_with_logits(
        logits, targets.float(), reduction="none"
    )
    n = valid.sum().clamp_min(1.0)
    return (raw * valid).sum() / n


# ---------------- Lightning module ----------------

class FloodLightningModule(pl.LightningModule):
    """Trains an adapter end-to-end on the binary SAR flood task."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        self.cfg = cfg
        self.model = self._build_model(cfg)
        # Optional gradient checkpointing toggled via env, used as an OOM-recovery
        # path in run_extras.sh when a config crashes with CUDA OOM.
        import os
        if os.environ.get("TRAINER_ENABLE_GRAD_CKPT") == "1":
            try:
                self.model.backbone.gradient_checkpointing_enable()
                print("[train] gradient_checkpointing enabled via TRAINER_ENABLE_GRAD_CKPT=1")
            except Exception as e:
                print(f"[train] could not enable gradient_checkpointing: {e}")
        self.iou_val = BinaryJaccardIndex(ignore_index=255)
        self.f1_val = BinaryF1Score(ignore_index=255)
        self.pr_val = BinaryPrecision(ignore_index=255)
        self.rc_val = BinaryRecall(ignore_index=255)
        # populated externally by the trainer at end of fit
        self.test_results: dict = {}

    @staticmethod
    def _build_model(cfg) -> nn.Module:
        """Build the adapter for the configured backbone.

        SAM and SAM 2 variants map to HuggingFace model identifiers and
        share the same SAMAdapter/SAM2Adapter wrappers. The larger
        variants (ViT-L, ViT-H, Hiera-Large) exceed the 12 GB VRAM of the
        consumer RTX 3060 and are intended for the optional Vast.ai 4090
        sweep documented in CONTEXT.md.
        """
        m = cfg.model
        SAM_IDS = {
            "sam-vit-b": "facebook/sam-vit-base",
            "sam-vit-l": "facebook/sam-vit-large",
            "sam-vit-h": "facebook/sam-vit-huge",
        }
        SAM2_IDS = {
            "sam2-hiera-t":  "facebook/sam2-hiera-tiny",
            "sam2-hiera-s":  "facebook/sam2-hiera-small",
            "sam2-hiera-bp": "facebook/sam2-hiera-base-plus",
            "sam2-hiera-l":  "facebook/sam2-hiera-large",
        }
        if m.backbone in SAM_IDS:
            return SAMAdapter(
                pretrained_id=SAM_IDS[m.backbone],
                peft_method=m.peft, rank=m.rank, adaptformer_dim=m.adaptformer_dim,
                mask_decoder_strategy=m.mask_decoder_strategy,
            )
        if m.backbone in SAM2_IDS:
            return SAM2Adapter(
                pretrained_id=SAM2_IDS[m.backbone],
                peft_method=m.peft, rank=m.rank, adaptformer_dim=m.adaptformer_dim,
                mask_decoder_strategy=m.mask_decoder_strategy,
                use_memory=m.use_memory,
            )
        raise ValueError(f"Unknown backbone: {m.backbone!r}")

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        images = batch["image"]
        labels = batch["label"]
        valid = (labels != 255).float()
        # Convert ignore (255) to a benign value so loss masking does the work.
        targets = torch.where(labels == 255, torch.zeros_like(labels), labels)
        logits = self.model(images)
        loss_d = dice_loss(logits, targets, valid)
        loss_b = bce_loss(logits, targets, valid)
        w = self.cfg.train.loss
        loss = w.dice_weight * loss_d + w.bce_weight * loss_b
        return loss, logits, labels

    def training_step(self, batch, _):
        loss, _, _ = self._shared_step(batch)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        loss, logits, labels = self._shared_step(batch)
        preds = (torch.sigmoid(logits) > 0.5).int()
        # torchmetrics expects (preds, target). Replace 255 ignore -> -1 so
        # ignore_index works for BinaryJaccardIndex which interprets 255 OK.
        self.iou_val.update(preds, labels.int())
        self.f1_val.update(preds, labels.int())
        self.pr_val.update(preds, labels.int())
        self.rc_val.update(preds, labels.int())
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        self.log("val/iou", self.iou_val.compute(), prog_bar=True)
        self.log("val/f1", self.f1_val.compute(), prog_bar=False)
        self.log("val/precision", self.pr_val.compute(), prog_bar=False)
        self.log("val/recall", self.rc_val.compute(), prog_bar=False)
        for m in (self.iou_val, self.f1_val, self.pr_val, self.rc_val):
            m.reset()

    def configure_optimizers(self):
        cfg = self.cfg.train.optimizer
        adapter_params = []
        unfrozen_params = []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            # Pretrained mask decoder params, when set to full-ft, are
            # "unfrozen" and get the lower LR. Adapter params (LoRA-like) get
            # the higher LR. Heuristic: anything containing "lora", "dora",
            # "adaptformer", or "magnitude" is treated as an adapter.
            if any(k in name for k in ("lora_", "magnitude", "adaptformer", "conv.weight")):
                adapter_params.append(p)
            else:
                unfrozen_params.append(p)

        param_groups = []
        if adapter_params:
            param_groups.append({"params": adapter_params, "lr": cfg.lr_adapter})
        if unfrozen_params:
            param_groups.append({"params": unfrozen_params, "lr": cfg.lr_unfrozen})
        opt = torch.optim.AdamW(
            param_groups, weight_decay=cfg.weight_decay, betas=tuple(cfg.betas)
        )
        # Cosine schedule with linear warmup.
        sched_cfg = self.cfg.train.scheduler
        total_steps = self.trainer.estimated_stepping_batches
        warmup = max(1, int(sched_cfg.warmup_steps))

        def lr_lambda(step):
            if step < warmup:
                return step / warmup
            progress = (step - warmup) / max(1, total_steps - warmup)
            import math
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step"}}


# ---------------- data loaders ----------------

def make_loaders(cfg):
    root = cfg.dataset.sen1floods11_root
    polari = cfg.dataset.polarimetric_mode
    train = Sen1Floods11Dataset(root=root, split="train", polarimetric_mode=polari)
    valid = Sen1Floods11Dataset(root=root, split="valid", polarimetric_mode=polari)
    nw = cfg.train.num_workers
    train_dl = DataLoader(
        train, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=nw, pin_memory=True, persistent_workers=nw > 0,
    )
    val_dl = DataLoader(
        valid, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=nw, pin_memory=True, persistent_workers=nw > 0,
    )
    return train_dl, val_dl


# ---------------- entry point ----------------

def load_cfg(path: Path):
    """Load a YAML config and merge its _base reference if present."""
    cfg = OmegaConf.load(path)
    base_path = path.parent / "_base.yaml"
    if base_path.exists() and path.name != "_base.yaml":
        base = OmegaConf.load(base_path)
        cfg = OmegaConf.merge(base, cfg)
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=None, help="Override finetune_epochs.")
    p.add_argument("--sen1floods11-root", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--max-steps", type=int, default=None,
                   help="Hard cap on training steps (useful for time-boxed runs).")
    p.add_argument("--limit-train-batches", type=float, default=None,
                   help="Fraction of train set to use per epoch (0.0-1.0).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config)
    if args.epochs is not None:
        cfg.train.finetune_epochs = args.epochs
    if args.sen1floods11_root is not None:
        cfg.dataset.sen1floods11_root = str(args.sen1floods11_root)
    output_dir = Path(args.output_dir) if args.output_dir is not None else Path(cfg.logging.output_dir)
    run_name = args.config.stem
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    pl.seed_everything(cfg.train.seed, workers=True)

    module = FloodLightningModule(cfg)
    train_dl, val_dl = make_loaders(cfg)

    ckpt = ModelCheckpoint(
        dirpath=str(run_dir),
        filename="best-{epoch:02d}-{val/iou:.3f}",
        monitor="val/iou", mode="max", save_top_k=1, save_last=True,
        auto_insert_metric_name=False,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.train.finetune_epochs,
        accumulate_grad_batches=cfg.train.grad_accum,
        precision=cfg.train.precision,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=10,
        callbacks=[ckpt],
        default_root_dir=str(run_dir),
        enable_progress_bar=True,
        max_steps=args.max_steps if args.max_steps is not None else -1,
        limit_train_batches=args.limit_train_batches if args.limit_train_batches is not None else 1.0,
    )

    print(f"[train] run_name={run_name}")
    print(f"[train] trainable params: {module.model.trainable_parameters()/1e6:.3f}M "
          f"({module.model.trainable_percentage():.2f}% of total)")
    trainer.fit(module, train_dl, val_dl)

    # Write a small JSON summary next to the checkpoints.
    summary = {
        "run_name": run_name,
        "config_path": str(args.config),
        "trainable_params": module.model.trainable_parameters(),
        "trainable_percentage": module.model.trainable_percentage(),
        "best_ckpt": str(ckpt.best_model_path) if ckpt.best_model_path else None,
        "best_val_iou": float(ckpt.best_model_score) if ckpt.best_model_score is not None else None,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[train] best val IoU: {summary['best_val_iou']}")
    print(f"[train] summary written to {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
