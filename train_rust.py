# train_rust_autosplit.py
import argparse, os, json, random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
import timm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

random.seed(42)

SUPPORTED_EXTS = {".jpg",".jpeg",".png",".ppm",".bmp",".pgm",".tif",".tiff",".webp"}

def has_images(root: Path) -> bool:
    if not root.exists(): return False
    for sub in root.rglob("*"):
        if sub.is_file() and sub.suffix.lower() in SUPPORTED_EXTS:
            return True
    return False

def build_transforms(img_size=256, aug="strong"):
    if aug == "strong":
        train_tfms = transforms.Compose([
            transforms.Resize(int(img_size*1.2)),
            transforms.RandomResizedCrop(img_size, scale=(0.6,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.3,0.3,0.2,0.03)], p=0.7),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    else:
        train_tfms = transforms.Compose([
            transforms.Resize(int(img_size*1.1)),
            transforms.RandomResizedCrop(img_size, scale=(0.8,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    val_tfms = transforms.Compose([
        transforms.Resize(int(img_size*1.1)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return train_tfms, val_tfms

class ListDataset(Dataset):
    """Minimal dataset from a list of (path, class_idx)."""
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        path, y = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, y

def make_loaders_autosplit(data_dir, img_size, batch_size, workers, aug, val_ratio):
    train_tfms, val_tfms = build_transforms(img_size, aug)
    train_dir = Path(data_dir)/"train"
    val_dir   = Path(data_dir)/"val"

    # If val has images, use normal ImageFolder loaders
    if has_images(val_dir):
        train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
        val_ds   = datasets.ImageFolder(val_dir,   transform=val_tfms)
        class_names = train_ds.classes
        counts = [0]*len(class_names)
        for _, y in train_ds.samples: counts[y]+=1
        weights = torch.tensor([sum(counts)/max(c,1) for c in counts], dtype=torch.float)
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=workers, pin_memory=True),
            DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True),
            class_names, weights
        )

    # Else: auto-build val from train
    print("[INFO] No validation set found — auto-splitting from train with val_ratio =", val_ratio)
    full_ds = datasets.ImageFolder(train_dir)  # no transforms here
    class_names = full_ds.classes
    # build per-class lists
    per_cls = {i: [] for i in range(len(class_names))}
    for (path, y) in full_ds.samples:
        if Path(path).suffix.lower() in SUPPORTED_EXTS:
            per_cls[y].append(path)
    # stratified split
    train_samples, val_samples = [], []
    for y, paths in per_cls.items():
        random.shuffle(paths)
        n_val = max(1, int(len(paths)*val_ratio)) if len(paths) > 0 else 0
        val_paths   = paths[:n_val]
        train_paths = paths[n_val:]
        val_samples   += [(p, y) for p in val_paths]
        train_samples += [(p, y) for p in train_paths]

    # class weights (based on new train split)
    counts = [0]*len(class_names)
    for _, y in train_samples: counts[y]+=1
    weights = torch.tensor([sum(counts)/max(c,1) for c in counts], dtype=torch.float)

    train_ds = ListDataset(train_samples, transform=train_tfms)
    val_ds   = ListDataset(val_samples,   transform=val_tfms)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=workers, pin_memory=True),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True),
        class_names, weights
    )

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(1)
        ys.extend(y.cpu().tolist()); ps.extend(pred.cpu().tolist())
    acc = accuracy_score(ys, ps)
    if len(set(ys)) == 2:
        f1  = f1_score(ys, ps, average="binary")
        pre = precision_score(ys, ps, zero_division=0, average="binary")
        rec = recall_score(ys, ps, zero_division=0, average="binary")
    else:
        f1  = f1_score(ys, ps, average="macro")
        pre = precision_score(ys, ps, zero_division=0, average="macro")
        rec = recall_score(ys, ps, zero_division=0, average="macro")
    return {"accuracy":acc, "f1":f1, "precision":pre, "recall":rec}

def save_ckpt(path, model, model_name, class_names, img_size):
    torch.save({
        "model": model.state_dict(),
        "model_name": model_name,
        "class_names": class_names,
        "img_size": img_size
    }, path)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, class_names, class_weights = make_loaders_autosplit(
        args.data_dir, args.img_size, args.batch_size, args.workers, args.aug, args.val_ratio
    )

    model = timm.create_model(args.model, pretrained=True, num_classes=len(class_names))
    model.to(device)

    if args.freeze_backbone > 0:
        for n,p in model.named_parameters():
            if any(k in n for k in ["fc","classifier","head"]):
                p.requires_grad = True
            else:
                p.requires_grad = False

    crit  = nn.CrossEntropyLoss(weight=class_weights.to(device))
    opt   = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

    best_f1 = -1.0
    best_path = outdir / f"{args.model.replace('/','_')}_best.pth"
    history = []
    patience_left = args.patience

    for epoch in range(1, args.epochs+1):
        model.train()
        if epoch == args.freeze_backbone + 1:
            for p in model.parameters(): p.requires_grad = True

        running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=100)
        for x,y in pbar:
            x,y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                logits = model(x)
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            running += loss.item() * x.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        sched.step()

        val = evaluate(model, val_loader, device)
        train_loss = running/len(train_loader.dataset)
        row = {"epoch":epoch, "train_loss":train_loss, **val}
        history.append(row)
        print(f"val={val} | train_loss={train_loss:.4f}")

        if val["f1"] > best_f1 + 1e-4:
            best_f1 = val["f1"]
            save_ckpt(best_path, model, args.model, class_names, args.img_size)
            patience_left = args.patience
            print(f"✓ saved new best: {best_path} (F1={best_f1:.4f})")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

        with open(outdir/"history.json","w") as f:
            json.dump(history, f, indent=2)

    print("Done. Best checkpoint:", best_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser("Fine-tune ResNet (or any timm model) with optional auto-split val")
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--model", type=str, default="resnet50")
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--aug", type=str, choices=["light","strong"], default="strong")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--freeze-backbone", type=int, default=0)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--val-ratio", type=float, default=0.2, help="used when data/val is missing/empty")
    p.add_argument("--out", type=str, default="runs/rust_cls")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    main(args)
