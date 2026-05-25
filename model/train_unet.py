"""U-Net baseline training.

The PANGAEA benchmark showed that a vanilla U-Net often matches or
beats large foundation models on Sen1Floods11. The thesis must include
this baseline honestly. We use segmentation_models_pytorch (smp) for
the U-Net architecture rather than reinventing it; smp.Unet with a
ResNet-34 encoder pretrained on ImageNet has ~25 M parameters, which
is well below the PEFT methods' total but with comparable trainable
counts (since the PEFT methods only train 0.5-2%).

Reuses the same loss (dice + bce), optimizer (AdamW), schedule
(linear warmup + cosine), and metric stack (torchmetrics) as
model/train.py so the comparison is apples-to-apples.

CLI
---
    python -m model.train_unet --seed 42
    python -m model.train_unet --seed 123 --epochs 10
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.datasets.sen1floods11 import Sen1Floods11Dataset
from model.train import bce_loss, dice_loss, load_cfg


class UNetModule(pl.LightningModule):
    """U-Net with ResNet-34 encoder, 3-channel pseudo-RGB input, 1-channel logit output."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        self.cfg = cfg
        import segmentation_models_pytorch as smp
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )
        self.iou_val = BinaryJaccardIndex(ignore_index=255)
        self.f1_val = BinaryF1Score(ignore_index=255)
        self.pr_val = BinaryPrecision(ignore_index=255)
        self.rc_val = BinaryRecall(ignore_index=255)

    def forward(self, x):
        return self.model(x).squeeze(1)  # (B, 1, H, W) -> (B, H, W)

    def _shared(self, batch):
        images = batch["image"]
        labels = batch["label"]
        valid = (labels != 255).float()
        targets = torch.where(labels == 255, torch.zeros_like(labels), labels)
        logits = self.forward(images)
        loss_d = dice_loss(logits, targets, valid)
        loss_b = bce_loss(logits, targets, valid)
        w = self.cfg.train.loss
        loss = w.dice_weight * loss_d + w.bce_weight * loss_b
        return loss, logits, labels

    def training_step(self, batch, _):
        loss, _, _ = self._shared(batch)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        loss, logits, labels = self._shared(batch)
        preds = (torch.sigmoid(logits) > 0.5).int()
        self.iou_val.update(preds, labels.int())
        self.f1_val.update(preds, labels.int())
        self.pr_val.update(preds, labels.int())
        self.rc_val.update(preds, labels.int())
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)

    def on_validation_epoch_end(self):
        self.log("val/iou", self.iou_val.compute(), prog_bar=True)
        self.log("val/f1", self.f1_val.compute())
        self.log("val/precision", self.pr_val.compute())
        self.log("val/recall", self.rc_val.compute())
        for m in (self.iou_val, self.f1_val, self.pr_val, self.rc_val):
            m.reset()

    def configure_optimizers(self):
        cfg = self.cfg.train.optimizer
        # U-Net trains all params, so we just put them in one group at lr_adapter.
        opt = torch.optim.AdamW(
            self.parameters(), lr=cfg.lr_adapter, weight_decay=cfg.weight_decay,
            betas=tuple(cfg.betas),
        )
        sched_cfg = self.cfg.train.scheduler
        total_steps = self.trainer.estimated_stepping_batches
        warmup = max(1, int(sched_cfg.warmup_steps))

        import math

        def lr_lambda(step):
            if step < warmup:
                return step / warmup
            progress = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step"}}


def make_loaders(cfg):
    root = cfg.dataset.sen1floods11_root
    polari = cfg.dataset.polarimetric_mode
    train = Sen1Floods11Dataset(root=root, split="train", polarimetric_mode=polari)
    valid = Sen1Floods11Dataset(root=root, split="valid", polarimetric_mode=polari)
    nw = cfg.train.num_workers
    train_dl = DataLoader(train, batch_size=cfg.train.batch_size, shuffle=True,
                          num_workers=nw, pin_memory=True, persistent_workers=nw > 0)
    val_dl = DataLoader(valid, batch_size=cfg.train.batch_size, shuffle=False,
                        num_workers=nw, pin_memory=True, persistent_workers=nw > 0)
    return train_dl, val_dl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--sen1floods11-root", type=Path,
                   default=Path("./data/sen1floods11"))
    p.add_argument("--base-config", type=Path,
                   default=Path(__file__).parent / "configs" / "_base.yaml")
    p.add_argument("--output-dir", type=Path, default=Path("runs"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.base_config)
    cfg.dataset.sen1floods11_root = str(args.sen1floods11_root)
    cfg.train.seed = args.seed
    cfg.train.finetune_epochs = args.epochs

    run_name = f"unet_seed{args.seed}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    pl.seed_everything(args.seed, workers=True)
    module = UNetModule(cfg)
    train_dl, val_dl = make_loaders(cfg)

    ckpt = ModelCheckpoint(
        dirpath=str(run_dir), filename="best-{epoch:02d}-{val/iou:.3f}",
        monitor="val/iou", mode="max", save_top_k=1, save_last=True,
        auto_insert_metric_name=False,
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accumulate_grad_batches=cfg.train.grad_accum,
        precision=cfg.train.precision,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[ckpt],
        default_root_dir=str(run_dir),
        log_every_n_steps=10,
    )
    n_train = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"[unet] run_name={run_name}  trainable={n_train/1e6:.2f}M  epochs={args.epochs}")
    trainer.fit(module, train_dl, val_dl)

    summary = {
        "run_name": run_name,
        "trainable_params": n_train,
        "best_ckpt": str(ckpt.best_model_path) if ckpt.best_model_path else None,
        "best_val_iou": float(ckpt.best_model_score) if ckpt.best_model_score is not None else None,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[unet] best val IoU: {summary['best_val_iou']}")


if __name__ == "__main__":
    main()
