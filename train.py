"""
train.py
========
LCMS-DB Training Script.

Usage:
    python train.py --config configs/ablation_4_full.yaml
    python train.py --config configs/ablation_1_baseline.yaml --resume runs/ablation_1_baseline/last.pt

Config flags (in YAML) control which modules are active, enabling all 4 ablation
experiments from a single codebase with no code changes.
"""

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm

# -- Local imports --------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import build_dataloaders
from losses.yolo_loss import YOLOLoss
from models.lcms_db import LCMSDB, build_model, load_checkpoint

# -- Metric helpers -------------------------------------------------------------

def compute_map(predictions_list, targets_list, num_classes: int,
                resolution: int = 640, iou_thresh: float = 0.5) -> float:
    """
    mAP@IoU computation on decoded predictions across all batches.

    Args:
        predictions_list: list of decoded prediction tensors (B, N, 4+1+C) per batch
        targets_list: list of target tensors (N_boxes, 6) per batch
        num_classes: number of object classes
        resolution: inference resolution in pixels
        iou_thresh: IoU threshold for TP

    Returns:
        mean AP across all classes (scalar float)
    """
    from torchvision.ops import box_iou, nms

    # Collect per-class TP/FP/scores
    class_records: Dict[int, Dict] = {
        c: {'scores': [], 'tp': [], 'n_gt': 0} for c in range(num_classes)
    }

    def xywh_to_xyxy(b):
        x1 = b[..., 0] - b[..., 2] / 2
        y1 = b[..., 1] - b[..., 3] / 2
        x2 = b[..., 0] + b[..., 2] / 2
        y2 = b[..., 1] + b[..., 3] / 2
        return torch.stack([x1, y1, x2, y2], dim=-1)

    for preds, targets in zip(predictions_list, targets_list):
        if preds is None or len(preds) == 0:
            for tgt in targets:
                cls_id = int(tgt[1].item())
                if cls_id in class_records:
                    class_records[cls_id]['n_gt'] += 1
            continue

        for batch_idx in range(preds.shape[0]):
            batch_preds = preds[batch_idx]  # (N, 4+1+C)
            batch_targets = targets[targets[:, 0] == batch_idx]  # (M, 6)

            if batch_targets.shape[0] == 0:
                continue

            gt_classes = batch_targets[:, 1].long()
            # GT boxes: normalized xywh -> pixel xyxy
            gt_xywh_px = batch_targets[:, 2:] * resolution
            gt_xyxy = xywh_to_xyxy(gt_xywh_px)

            if batch_preds.shape[0] == 0:
                for c in gt_classes.tolist():
                    if c in class_records:
                        class_records[c]['n_gt'] += 1
                continue

            obj_scores = batch_preds[:, 4]
            cls_scores = batch_preds[:, 5:]
            scores, cls_ids = (obj_scores.unsqueeze(1) * cls_scores).max(dim=1)

            # Filter low confidence for speed and NMS quality
            conf_mask = scores >= 0.05
            if not conf_mask.any():
                for c in gt_classes.tolist():
                    if c in class_records:
                        class_records[c]['n_gt'] += 1
                continue

            b_preds = batch_preds[conf_mask]
            b_scores = scores[conf_mask]
            b_cls = cls_ids[conf_mask]

            pred_xyxy = xywh_to_xyxy(b_preds[:, :4])

            # Apply NMS (cast to float for AMP half-precision compatibility)
            keep = nms(pred_xyxy.float(), b_scores.float(), 0.45)
            pred_xyxy = pred_xyxy[keep]
            b_scores = b_scores[keep]
            b_cls = b_cls[keep]

            for c in range(num_classes):
                gt_mask   = (gt_classes == c)
                pred_mask = (b_cls == c)
                class_records[c]['n_gt'] += gt_mask.sum().item()

                if pred_mask.sum() == 0:
                    continue

                p_boxes = pred_xyxy[pred_mask]
                p_scores = b_scores[pred_mask]
                g_boxes  = gt_xyxy[gt_mask]

                matched_gt = set()
                for pb, ps in zip(p_boxes, p_scores):
                    class_records[c]['scores'].append(ps.item())
                    if g_boxes.shape[0] == 0:
                        class_records[c]['tp'].append(0)
                        continue
                    ious = box_iou(pb.unsqueeze(0), g_boxes)[0]
                    best_iou, best_gi = ious.max(0)
                    if best_iou >= iou_thresh and best_gi.item() not in matched_gt:
                        class_records[c]['tp'].append(1)
                        matched_gt.add(best_gi.item())
                    else:
                        class_records[c]['tp'].append(0)

    # Compute AP per class via 11-point interpolation
    aps = []
    for c in range(num_classes):
        rec = class_records[c]
        n_gt = rec['n_gt']
        if n_gt == 0:
            continue
        scores_arr = rec['scores']
        tp_arr = rec['tp']
        if len(scores_arr) == 0:
            aps.append(0.0)
            continue

        order = sorted(range(len(scores_arr)), key=lambda i: -scores_arr[i])
        tp_sorted = [tp_arr[i] for i in order]
        cum_tp = 0
        cum_fp = 0
        precisions = []
        recalls    = []
        for tp_val in tp_sorted:
            cum_tp += tp_val
            cum_fp += 1 - tp_val
            precisions.append(cum_tp / (cum_tp + cum_fp + 1e-9))
            recalls.append(cum_tp / (n_gt + 1e-9))

        # 11-point interpolation
        ap = 0.0
        for t in [i / 10 for i in range(11)]:
            prec_at_t = [p for p, r in zip(precisions, recalls) if r >= t]
            ap += max(prec_at_t) if prec_at_t else 0.0
        ap /= 11.0
        aps.append(ap)

    return float(sum(aps) / len(aps)) if aps else 0.0


# -- Config loading -------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """
    Load YAML config, supporting a simple ``_base_: other.yaml`` inheritance.

    Args:
        config_path: path to the YAML config file

    Returns:
        merged config dict
    """
    config_path = Path(config_path).resolve()
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    if '_base_' in cfg:
        base_name = cfg.pop('_base_')
        base_path = config_path.parent / base_name
        base_cfg = load_config(str(base_path))
        base_cfg = deep_merge(base_cfg, cfg)
        return base_cfg
    return cfg


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# -- Training utilities ---------------------------------------------------------

def build_optimizer(model: nn.Module, cfg: dict):
    """Build SGD or Adam optimizer from config."""
    lr   = cfg['train']['learning_rate']
    wd   = cfg['train']['weight_decay']
    mom  = cfg['train'].get('momentum', 0.937)

    # Separate params: bias/BN (no weight decay) vs. rest
    decay_params    = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or 'bn' in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return SGD(
        [
            {'params': decay_params,    'weight_decay': wd},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ],
        lr=lr,
        momentum=mom,
        nesterov=True,
    )


def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    """Build LR scheduler from config."""
    sched_type = cfg['train'].get('scheduler', 'cosine')
    epochs     = cfg['train']['epochs']
    warmup_ep  = cfg['train'].get('warmup_epochs', 3)

    if sched_type == 'cosine':
        # Main scheduler (after warmup)
        main_scheduler = CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_ep, eta_min=1e-6
        )
    else:
        main_scheduler = StepLR(
            optimizer,
            step_size=cfg['train'].get('step_size', 30),
            gamma=cfg['train'].get('step_gamma', 0.1),
        )
    return main_scheduler, warmup_ep


def warmup_lr(optimizer, epoch: int, step: int, steps_per_epoch: int,
              warmup_epochs: int, base_lr: float, warmup_lr_start: float):
    """Linear warmup learning rate."""
    total_warmup_steps = warmup_epochs * steps_per_epoch
    current_step = epoch * steps_per_epoch + step
    if current_step < total_warmup_steps:
        lr = warmup_lr_start + (base_lr - warmup_lr_start) * (
            current_step / max(1, total_warmup_steps)
        )
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        return True
    return False


# -- Training / Validation loops ------------------------------------------------

def train_one_epoch(
    model: LCMSDB,
    loader,
    optimizer,
    criterion: YOLOLoss,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    cfg: dict,
    steps_per_epoch: int,
    main_scheduler,
    warmup_epochs: int,
) -> Dict[str, float]:
    """
    Run one training epoch.

    Returns:
        dict of mean loss components: {'total', 'box', 'cls', 'obj'}
    """
    model.train()
    resolution   = cfg['resolution']
    grad_clip    = cfg['train'].get('grad_clip', 10.0)
    log_interval = cfg['train'].get('log_interval', 10)
    base_lr      = cfg['train']['learning_rate']
    warmup_lr_v  = cfg['train'].get('warmup_lr', 1e-4)
    use_amp      = cfg['train'].get('amp', True) and device.type == 'cuda'

    running = {'total': 0.0, 'box': 0.0, 'cls': 0.0, 'obj': 0.0}
    n_batches = len(loader)

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [train]", leave=False, unit='batch')
    for step, batch in enumerate(pbar):
        # -- Warmup LR --
        in_warmup = warmup_lr(
            optimizer, epoch, step, steps_per_epoch,
            warmup_epochs, base_lr, warmup_lr_v
        )
        if not in_warmup and step == 0 and epoch > warmup_epochs:
            main_scheduler.step()

        images  = batch['images'].to(device, non_blocking=True)
        targets = batch['labels'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            preds = model(images)
            loss, loss_dict = criterion(preds, targets, resolution)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        # -- Accumulate metrics --
        running['total'] += loss.item()
        for k in ('box', 'cls', 'obj'):
            running[k] += loss_dict.get(k, 0.0)

        if (step + 1) % log_interval == 0:
            pbar.set_postfix(
                loss=f"{running['total']/(step+1):.4f}",
                box=f"{running['box']/(step+1):.4f}",
                cls=f"{running['cls']/(step+1):.4f}",
            )

    return {k: v / max(1, n_batches) for k, v in running.items()}


@torch.no_grad()
def validate(
    model: LCMSDB,
    loader,
    criterion: YOLOLoss,
    device: torch.device,
    cfg: dict,
) -> Dict[str, float]:
    """
    Run validation epoch.

    Returns:
        dict with 'val_loss', 'val_map50'
    """
    model.eval()
    resolution = cfg['resolution']
    use_amp    = cfg['train'].get('amp', True) and device.type == 'cuda'
    nc         = cfg['model']['num_classes']

    total_loss   = 0.0
    n_batches    = len(loader)
    preds_list   = []
    targets_list = []

    pbar = tqdm(loader, desc="  [val]", leave=False, unit='batch')
    for batch in pbar:
        images  = batch['images'].to(device, non_blocking=True)
        targets = batch['labels'].to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            preds = model(images)
            loss, _ = criterion(preds, targets, resolution)

        total_loss += loss.item()
        # Decode predictions for mAP
        decoded = model.head.decode_predictions(preds, resolution)
        preds_list.append(decoded.cpu())
        targets_list.append(targets.cpu())

    val_map50 = compute_map(
        preds_list,
        targets_list,
        num_classes=nc,
        resolution=resolution,
        iou_thresh=0.5,
    )

    return {
        'val_loss': total_loss / max(1, n_batches),
        'val_map50': val_map50,
    }


# -- Checkpointing --------------------------------------------------------------

def save_checkpoint(
    model: LCMSDB,
    optimizer,
    epoch: int,
    metrics: dict,
    save_dir: Path,
    filename: str = 'last.pt',
    cfg: dict = None,
):
    """Save model checkpoint to disk."""
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'metrics': metrics,
        'config': cfg,
    }
    torch.save(checkpoint, save_dir / filename)


def save_csv_row(csv_path: Path, row: dict):
    """Append one row to a CSV log file, writing header if needed."""
    write_header = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# -- Main -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train LCMS-DB')
    parser.add_argument('--config',   required=True,         help='Path to YAML config')
    parser.add_argument('--resume',   default=None,          help='Path to checkpoint to resume from')
    parser.add_argument('--device',   default='',            help='cuda:0 / cpu (auto if empty)')
    parser.add_argument('--epochs',   type=int, default=None, help='Override epochs from config')
    parser.add_argument('--batch',    type=int, default=None, help='Override batch_size from config')
    parser.add_argument('--seed',     type=int, default=None, help='Random seed (overrides train.seed in config)')
    args = parser.parse_args()

    # -- Config --
    cfg = load_config(args.config)
    if args.epochs:
        cfg['train']['epochs'] = args.epochs
    if args.batch:
        cfg['train']['batch_size'] = args.batch
    if args.seed is not None:
        cfg['train']['seed'] = args.seed

    # rev. 4: seeding. Python, NumPy and torch are seeded here; DataLoader workers
    # derive their Python/NumPy/torch seeds from the torch seed, so augmentation
    # and the class-balanced sampler are reproducible too. cuDNN kernels are not
    # forced to be deterministic, so runs are seeded but not bit-identical.
    seed = cfg['train'].get('seed', None)
    if seed is not None:
        import random as _random
        import numpy as _np
        _random.seed(seed); _np.random.seed(seed)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        print(f"[train] Seed: {seed}")
    else:
        print("[train] Seed: none (unseeded run)")

    save_dir = Path(cfg['train']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config next to checkpoints for reproducibility
    with open(save_dir / 'config.yaml', 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # -- Device --
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"[train] Device: {device}")

    # -- Data --
    train_loader, val_loader, _ = build_dataloaders(cfg)
    steps_per_epoch = len(train_loader)
    print(f"[train] Train batches: {steps_per_epoch} | Val batches: {len(val_loader)}")

    # -- Model --
    model = build_model(cfg).to(device)
    info  = model.get_info()
    print(f"[train] Total params:     {info['total_params']:,}")
    print(f"[train] Trainable params: {info['trainable_params']:,}")

    # -- Loss, Optimizer, Scheduler --
    criterion = YOLOLoss(cfg, str(device)).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler, warmup_epochs = build_scheduler(optimizer, cfg, steps_per_epoch)
    scaler = GradScaler(enabled=cfg['train'].get('amp', True) and device.type == 'cuda')

    # -- Resume --
    start_epoch = 0
    best_map    = 0.0
    if args.resume:
        model, ckpt_meta = load_checkpoint(model, args.resume, str(device))
        start_epoch = ckpt_meta.get('epoch', 0) + 1
        # rev. 2: continue the run exactly where it stopped -------------------
        # (a) optimizer state (SGD momentum buffers)
        if ckpt_meta.get('optimizer_state') is not None:
            try:
                optimizer.load_state_dict(ckpt_meta['optimizer_state'])
                # LR is re-derived below: start from the value left by the warmup
                # (its last step stops just short of the base LR), as in an
                # uninterrupted run.
                tw = max(1, warmup_epochs * steps_per_epoch)
                w0 = cfg['train'].get('warmup_lr', 1e-4)
                for pg in optimizer.param_groups:
                    base = pg.get('initial_lr', pg['lr'])
                    pg['lr'] = w0 + (base - w0) * ((tw - 1) / tw) if warmup_epochs > 0 else base
                print("[train] Optimizer state restored")
            except Exception as e:
                print(f"[train] WARNING: optimizer state not restored ({e})")
        # (b) fast-forward the cosine schedule: train_one_epoch steps it once at
        #     the start of every epoch e > warmup_epochs
        n_sched = sum(1 for e in range(start_epoch) if e > warmup_epochs)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for _ in range(n_sched):
                scheduler.step()
        # (c) best score so far = the one stored in best.pt (not the last epoch's)
        best_map = ckpt_meta.get('metrics', {}).get('val_map50', 0.0)
        best_path = save_dir / 'best.pt'
        if best_path.exists():
            try:
                best_map = torch.load(best_path, map_location='cpu', weights_only=False)['metrics']['val_map50']
            except Exception:
                pass
        print(f"[train] Resumed at epoch {start_epoch + 1}, best mAP@0.5 so far={best_map:.4f}, "
              f"scheduler advanced {n_sched} steps, lr={optimizer.param_groups[0]['lr']:.6f}")

    # -- Training Loop --
    patience     = cfg['train'].get('early_stopping_patience', 20)
    no_improve   = 0
    epochs       = cfg['train']['epochs']
    csv_log_path = save_dir / 'results.csv'

    print(f"\n[train] Starting training for {epochs} epochs...\n")

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        # -- Train --
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler,
            device, epoch, cfg, steps_per_epoch, scheduler, warmup_epochs,
        )

        # -- Validate --
        val_metrics = validate(model, val_loader, criterion, device, cfg)

        epoch_time = time.time() - t0
        metrics    = {**train_metrics, **val_metrics, 'epoch': epoch + 1, 'time': epoch_time}

        # -- Logging --
        print(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"loss={train_metrics['total']:.4f} "
            f"(box={train_metrics['box']:.4f}, "
            f"cls={train_metrics['cls']:.4f}, "
            f"obj={train_metrics['obj']:.4f}) | "
            f"val_loss={val_metrics['val_loss']:.4f} | "
            f"val_mAP@0.5={val_metrics['val_map50']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.6f} | "
            f"t={epoch_time:.1f}s"
        )
        save_csv_row(csv_log_path, metrics)

        # -- Checkpointing --
        save_checkpoint(model, optimizer, epoch, metrics, save_dir, 'last.pt', cfg)
        if val_metrics['val_map50'] > best_map:
            best_map = val_metrics['val_map50']
            no_improve = 0
            save_checkpoint(model, optimizer, epoch, metrics, save_dir, 'best.pt', cfg)
            print(f"  [OK] New best mAP@0.5 = {best_map:.4f} -> saved best.pt")
        else:
            no_improve += 1

        # -- Early stopping --
        if no_improve >= patience:
            print(f"\n[train] Early stopping: no improvement for {patience} epochs.")
            break

    print(f"\n[train] Done. Best mAP@0.5 = {best_map:.4f}")
    print(f"[train] Checkpoints saved to: {save_dir}")


if __name__ == '__main__':
    main()
