import argparse
import os
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import set_seed

def build_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])
    return train_tf, eval_tf, eval_tf

def get_datasets(data_dir: Path, img_size: int):
    train_tf, val_tf, test_tf = build_transforms(img_size)
    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_ds   = datasets.ImageFolder(data_dir / "val",   transform=val_tf)
    test_ds  = datasets.ImageFolder(data_dir / "test",  transform=test_tf)
    return train_ds, val_ds, test_ds

def make_sampler_for_imbalance(train_ds: datasets.ImageFolder):
    # Compute class counts
    targets = np.array([y for _, y in train_ds.samples])
    class_sample_counts = np.bincount(targets)
    class_weights = 1.0 / np.maximum(class_sample_counts, 1)
    weights = class_weights[targets]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

def build_model(model_name: str = "efficientnet_b0", num_classes: int = 1):
    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        backbone = models.efficientnet_b0(weights=weights)
        in_feats = backbone.classifier[1].in_features
        backbone.classifier[1] = nn.Identity()
        head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_feats, num_classes)
        )
        model = nn.Sequential(backbone, head)
    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, num_classes)
    else:
        raise ValueError("Unsupported model. Choose 'efficientnet_b0' or 'resnet50'.")
    return model

def train_one_epoch(model, loader, device, criterion, optimizer, scaler):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.float().to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        preds = (torch.sigmoid(logits) >= 0.5).long()
        running_loss += loss.item() * images.size(0)
        correct += (preds.cpu() == labels.long().cpu()).sum().item()
        total += images.size(0)
    return running_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    y_true, y_prob = [], []
    for images, labels in tqdm(loader, desc="Eval", leave=False):
        images, labels = images.to(device), labels.float().to(device)
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()

        running_loss += loss.item() * images.size(0)
        correct += (preds.cpu() == labels.long().cpu()).sum().item()
        total += images.size(0)
        y_true.extend(labels.cpu().numpy().tolist())
        y_prob.extend(probs.cpu().numpy().tolist())

    avg_loss = running_loss / total
    acc = correct / total
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    return avg_loss, acc, auc, np.array(y_true), np.array(y_prob)

def plot_confusion_matrix(y_true, y_prob, out_path: Path):
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    classes = ["no_rust", "rust"]
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes, ylabel="True", xlabel="Pred")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def freeze_backbone(model, freeze=True):
    for name, param in model.named_parameters():
        if "classifier" in name or name.endswith(".fc.weight") or name.endswith(".fc.bias"):
            param.requires_grad = True
        else:
            param.requires_grad = not freeze

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True, help="Path to data/ with train/val/test/")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model_name", type=str, default="efficientnet_b0", choices=["efficientnet_b0", "resnet50"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5, help="early stopping patience (val loss)")
    parser.add_argument("--warmup_epochs", type=int, default=2, help="train head only for first N epochs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Datasets & loaders
    train_ds, val_ds, test_ds = get_datasets(args.data_dir, args.img_size)
    class_to_idx: Dict[str,int] = train_ds.class_to_idx
    assert set(class_to_idx.keys()) == {"rust", "no_rust"} or len(class_to_idx)==2, \
        f"Expected two classes (rust, no_rust). Got: {class_to_idx}"

    sampler = make_sampler_for_imbalance(train_ds)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = build_model(args.model_name).to(device)

    # Loss (BCEWithLogits for binary)
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    best_val_loss = float("inf")
    patience_ctr = 0

    # Warmup: train head only
    freeze_backbone(model, freeze=True)

    for epoch in range(1, args.epochs + 1):
        if epoch == args.warmup_epochs + 1:
            # unfreeze backbone for fine-tuning
            freeze_backbone(model, freeze=False)

        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, device, criterion, optimizer, scaler)
        val_loss, val_acc, val_auc, y_true_val, y_prob_val = evaluate(model, val_loader, device, criterion)
        scheduler.step()

        print(f"Train   | loss={train_loss:.4f} acc={train_acc:.4f}")
        print(f"Val     | loss={val_loss:.4f} acc={val_acc:.4f} auc={val_auc:.4f}")

        # Save checkpoint every epoch
        ckpt_path = args.out_dir / f"epoch_{epoch:03d}.pt"
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "class_to_idx": class_to_idx,
            "img_size": args.img_size,
            "model_name": args.model_name,
        }, ckpt_path)

        # Early stopping on val loss
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_ctr = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "class_to_idx": class_to_idx,
                "img_size": args.img_size,
                "model_name": args.model_name,
            }, args.out_dir / "best.pt")
            plot_confusion_matrix(y_true_val, y_prob_val, args.out_dir / "val_confusion_matrix.png")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    # Final test evaluation (best checkpoint)
    print("\nLoading best model for test evaluation...")
    best = torch.load(args.out_dir / "best.pt", map_location=device)
    model = build_model(best["model_name"]).to(device)
    model.load_state_dict(best["model_state"])

    test_loss, test_acc, test_auc, y_true_test, y_prob_test = evaluate(model, test_loader, device, criterion)
    print(f"Test    | loss={test_loss:.4f} acc={test_acc:.4f} auc={test_auc:.4f}")

    # Classification report at 0.5 threshold
    y_pred_test = (y_prob_test >= 0.5).astype(int)
    target_names = ["no_rust", "rust"]
    print("\nClassification report (threshold=0.5):")
    print(classification_report(y_true_test, y_pred_test, target_names=target_names))

    plot_confusion_matrix(y_true_test, y_prob_test, args.out_dir / "test_confusion_matrix.png")

if __name__ == "__main__":
    main()
