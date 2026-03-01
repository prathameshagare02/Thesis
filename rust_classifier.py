# rust_classifier.py
# One-file PyTorch pipeline for "rust" vs "no_rust":
# - train: fine-tune a pretrained model
# - eval:  evaluate on a test folder
# - infer: predict a single image
# - app:   launch a Gradio upload app

import argparse
import os
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from PIL import Image
from tqdm import tqdm

# Optional (only used in `app`)
try:
    import gradio as gr
except Exception:
    gr = None

# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
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
    return train_tf, eval_tf

def get_datasets(root: Path, img_size: int):
    train_tf, eval_tf = build_transforms(img_size)
    train_ds = datasets.ImageFolder(root / "train", transform=train_tf)
    val_ds   = datasets.ImageFolder(root / "val",   transform=eval_tf)
    test_ds  = datasets.ImageFolder(root / "test",  transform=eval_tf)
    return train_ds, val_ds, test_ds

def make_sampler_for_imbalance(train_ds: datasets.ImageFolder):
    targets = np.array([y for _, y in train_ds.samples])
    class_sample_counts = np.bincount(targets)
    class_weights = 1.0 / np.maximum(class_sample_counts, 1)
    weights = class_weights[targets]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

def build_model(model_name: str = "efficientnet_b0"):
    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        backbone = models.efficientnet_b0(weights=weights)
        in_feats = backbone.classifier[1].in_features
        backbone.classifier[1] = nn.Identity()
        head = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_feats, 1))
        model = nn.Sequential(backbone, head)
    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, 1)
    else:
        raise ValueError("Unsupported model. Use 'efficientnet_b0' or 'resnet50'.")
    return model

def freeze_backbone(model: nn.Module, freeze: bool = True):
    for name, p in model.named_parameters():
        # keep classifier/trainable head
        if ("classifier" in name) or name.endswith(".fc.weight") or name.endswith(".fc.bias"):
            p.requires_grad = True
        else:
            p.requires_grad = not (not freeze) if False else not freeze  # clearer below line:
    for name, p in model.named_parameters():
        if ("classifier" in name) or name.endswith(".fc.weight") or name.endswith(".fc.bias"):
            p.requires_grad = True
        else:
            p.requires_grad = not freeze

# -----------------------------
# Train / Eval helpers
# -----------------------------
def train_one_epoch(model, loader, device, criterion, optimizer, scaler):
    model.train()
    run_loss, correct, total = 0.0, 0, 0
    for x, y in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(device), y.float().to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logit = model(x).squeeze(1)
            loss = criterion(logit, y)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        prob = torch.sigmoid(logit)
        pred = (prob >= 0.5).long()
        run_loss += loss.item() * x.size(0)
        correct += (pred.cpu() == y.long().cpu()).sum().item()
        total += x.size(0)
    return run_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    run_loss, correct, total = 0.0, 0, 0
    y_true, y_prob = [], []
    for x, y in tqdm(loader, desc="Eval", leave=False):
        x, y = x.to(device), y.float().to(device)
        logit = model(x).squeeze(1)
        loss = criterion(logit, y)
        prob = torch.sigmoid(logit)
        pred = (prob >= 0.5).long()
        run_loss += loss.item() * x.size(0)
        correct += (pred.cpu() == y.long().cpu()).sum().item()
        total += x.size(0)
        y_true.extend(y.cpu().numpy().tolist())
        y_prob.extend(prob.cpu().numpy().tolist())
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    return run_loss / total, correct / total, auc, y_true, y_prob

def save_ckpt(path: Path, model, optimizer, epoch, class_to_idx, img_size, model_name):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "class_to_idx": class_to_idx,
        "img_size": img_size,
        "model_name": model_name,
    }, path)

def load_ckpt(ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(ckpt["model_name"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt

# -----------------------------
# Modes
# -----------------------------
def mode_train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds, test_ds = get_datasets(args.data_dir, args.img_size)

    # sanity for binary classes
    assert len(train_ds.classes) == 2, f"Need exactly 2 classes, got: {train_ds.classes}"
    class_to_idx: Dict[str, int] = train_ds.class_to_idx
    print(f"Classes: {class_to_idx}")

    sampler = make_sampler_for_imbalance(train_ds)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model(args.model_name).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    best_val_loss = float("inf")
    patience_ctr = 0

    # warmup: train head only
    freeze_backbone(model, freeze=True)

    for epoch in range(1, args.epochs + 1):
        if epoch == args.warmup_epochs + 1:
            freeze_backbone(model, freeze=False)

        print(f"\nEpoch {epoch}/{args.epochs}")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, criterion, optimizer, scaler)
        va_loss, va_acc, va_auc, y_true_val, y_prob_val = evaluate(model, val_loader, device, criterion)
        scheduler.step()

        print(f"Train | loss={tr_loss:.4f} acc={tr_acc:.4f}")
        print(f"Val   | loss={va_loss:.4f} acc={va_acc:.4f} auc={va_auc:.4f}")

        # save epoch ckpt
        save_ckpt(args.out_dir / f"epoch_{epoch:03d}.pt", model, optimizer, epoch, class_to_idx, args.img_size, args.model_name)

        # early stopping on val loss
        if va_loss + 1e-4 < best_val_loss:
            best_val_loss = va_loss
            patience_ctr = 0
            save_ckpt(args.out_dir / "best.pt", model, optimizer, epoch, class_to_idx, args.img_size, args.model_name)
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    # final test with best
    print("\nTesting best checkpoint…")
    model, ckpt = load_ckpt(args.out_dir / "best.pt", device)
    te_loss, te_acc, te_auc, y_true, y_prob = evaluate(model, test_loader, device, criterion)
    print(f"Test  | loss={te_loss:.4f} acc={te_acc:.4f} auc={te_auc:.4f}")

    y_pred = (y_prob >= 0.5).astype(int)
    target_names = [k for k,_ in sorted(ckpt["class_to_idx"].items(), key=lambda kv: kv[1])]
    print("\nClassification report (thr=0.5):")
    print(classification_report(y_true, y_pred, target_names=target_names))

    # simple confusion matrix print
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n", cm)

    # Print ROC threshold suggestion (Youden J)
    try:
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        j = tpr - fpr
        best_idx = int(j.argmax())
        print(f"Suggested threshold by Youden J: {thr[best_idx]:.3f} (TPR={tpr[best_idx]:.3f}, FPR={fpr[best_idx]:.3f})")
    except Exception:
        pass

def mode_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_ckpt(args.ckpt, device)
    img_size = ckpt.get("img_size", 224)
    _, eval_tf = build_transforms(img_size)

    test_ds = datasets.ImageFolder(Path(args.data_dir) / "test", transform=eval_tf)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    criterion = nn.BCEWithLogitsLoss()

    te_loss, te_acc, te_auc, y_true, y_prob = evaluate(model, test_loader, device, criterion)
    print(f"Test  | loss={te_loss:.4f} acc={te_acc:.4f} auc={te_auc:.4f}")

    y_pred = (y_prob >= 0.5).astype(int)
    target_names = [k for k,_ in sorted(ckpt["class_to_idx"].items(), key=lambda kv: kv[1])]
    print("\nClassification report (thr=0.5):")
    print(classification_report(y_true, y_pred, target_names=target_names))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

def mode_infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_ckpt(args.ckpt, device)
    img_size = ckpt.get("img_size", 224)
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])

    img = Image.open(args.image).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(x).squeeze(1)
        p_rust = torch.sigmoid(logit).item()

    # class mapping fallback
    class_to_idx = ckpt.get("class_to_idx", {"no_rust": 0, "rust": 1})
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    pred = 1 if p_rust >= args.threshold else 0
    print(f"Prediction: {idx_to_class[pred]} | P(rust)={p_rust:.3f} | threshold={args.threshold}")

def mode_app(args):
    if gr is None:
        raise RuntimeError("Gradio not installed. `pip install gradio`")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_ckpt(args.ckpt, device)
    img_size = ckpt.get("img_size", 224)
    class_to_idx = ckpt.get("class_to_idx", {"no_rust": 0, "rust": 1})
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])

    def predict(image: Image.Image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        x = tf(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logit = model(x).squeeze(1)
            p_rust = torch.sigmoid(logit).item()
        pred = 1 if p_rust >= args.threshold else 0
        return idx_to_class[pred], {"rust": p_rust, "no_rust": 1 - p_rust}

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="Upload image"),
        outputs=[gr.Label(label="Prediction"), gr.Label(label="Confidence")],
        title="Rust vs No Rust Classifier",
        description=f"Threshold = {args.threshold} (tune for your use-case)."
    )
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)

# -----------------------------
# CLI
# -----------------------------
def build_parser():
    p = argparse.ArgumentParser(description="Rust vs No-Rust classifier (single file).")
    sub = p.add_subparsers(dest="mode", required=True)

    # Shared defaults
    p.add_argument("--model_name", type=str, default="efficientnet_b0", choices=["efficientnet_b0", "resnet50"])
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)

    # train
    pt = sub.add_parser("train", help="Train and save best checkpoint")
    pt.add_argument("--data_dir", type=Path, required=True, help="data/ with train/ val/ test/")
    pt.add_argument("--out_dir", type=Path, default=Path("outputs"))
    pt.add_argument("--epochs", type=int, default=20)
    pt.add_argument("--batch_size", type=int, default=32)
    pt.add_argument("--lr", type=float, default=1e-4)
    pt.add_argument("--weight_decay", type=float, default=1e-4)
    pt.add_argument("--patience", type=int, default=5)
    pt.add_argument("--warmup_epochs", type=int, default=2)

    # eval
    pe = sub.add_parser("eval", help="Evaluate a checkpoint on test set")
    pe.add_argument("--data_dir", type=Path, required=True)
    pe.add_argument("--ckpt", type=Path, required=True)
    pe.add_argument("--batch_size", type=int, default=32)

    # infer
    pi = sub.add_parser("infer", help="Predict a single image")
    pi.add_argument("--ckpt", type=Path, required=True)
    pi.add_argument("--image", type=Path, required=True)
    pi.add_argument("--threshold", type=float, default=0.5)

    # app
    pa = sub.add_parser("app", help="Launch Gradio upload app")
    pa.add_argument("--ckpt", type=Path, required=True)
    pa.add_argument("--threshold", type=float, default=0.5)
    pa.add_argument("--port", type=int, default=7860)
    pa.add_argument("--share", action="store_true")

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "train":
        mode_train(args)
    elif args.mode == "eval":
        mode_eval(args)
    elif args.mode == "infer":
        mode_infer(args)
    elif args.mode == "app":
        mode_app(args)
    else:
        raise ValueError("Unknown mode")

if __name__ == "__main__":
    main()
