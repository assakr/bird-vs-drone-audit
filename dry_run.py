import sys
from pathlib import Path
import torch
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from train import load_config, build_optimizer
from data.dataset import build_dataloaders
from models.lcms_db import build_model
from losses.yolo_loss import YOLOLoss

def main():
    print("[dry_run] Loading configs/v2_full.yaml...")
    cfg = load_config("configs/v2_full.yaml")
    print("[dry_run] Building dataloaders...")
    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[dry_run] Building model on {device}...")
    model = build_model(cfg).to(device)
    info = model.get_info()
    print(f"[dry_run] LCMS-DB v2 Total params: {info['total_params']:,}")
    print(f"[dry_run] Model strides: {model.strides}")

    crit = YOLOLoss(cfg, str(device)).to(device)
    opt = build_optimizer(model, cfg)

    print("[dry_run] Fetching 1 train batch...")
    for batch in train_loader:
        imgs = batch['images'].to(device)
        tgts = batch['labels'].to(device)
        print(f"[dry_run] Images shape: {imgs.shape}, Targets count: {len(tgts)}")
        with torch.cuda.amp.autocast(enabled=True):
            preds = model(imgs)
            print(f"[dry_run] Predictions scales: {len(preds)}, P2 shape: {preds[0].shape}")
            loss, loss_dict = crit(preds, tgts, cfg['resolution'])
        print(f"[dry_run] Loss: {loss.item():.4f}, Dict: {loss_dict}")
        loss.backward()
        opt.step()
        print("[dry_run] Backward pass SUCCESS!")
        break

    print("[dry_run] Fetching 1 val batch...")
    for batch in val_loader:
        imgs = batch['images'].to(device)
        tgts = batch['labels'].to(device)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            preds = model(imgs)
            decoded = model.head.decode_predictions(preds, cfg['resolution'])
            print(f"[dry_run] Decoded shape: {decoded.shape}")
        print("[dry_run] Val decode SUCCESS!")
        break

    print("\n[dry_run] ALL CHECKS PASSED PERFECTLY!\n")

if __name__ == '__main__':
    main()
