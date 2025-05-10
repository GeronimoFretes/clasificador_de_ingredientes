import argparse, os, signal, sys, time
from pathlib import Path
import torch, torchvision as tv
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms as T
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from PIL import Image
import pandas as pd

class RGBImageFolder(ImageFolder):
    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data",         default="data/split", help="data root")
    p.add_argument("--epochs",       default=20,  type=int)
    p.add_argument("--batch",        default=32,  type=int)
    p.add_argument("--img_size",     default=224, type=int)
    p.add_argument("--lr",           default=1e-4, type=float)
    p.add_argument("--workers",      default=os.cpu_count()//2, type=int)
    p.add_argument("--amp",          action="store_true", help="mixed precision")
    p.add_argument("--checkpoint",   default="models/best_model.pth")
    return p.parse_args()

# ──────────────────────────────────────────────────────────────────────────────
def build_loaders(root: Path, img_size: int, batch: int, n_workers: int
                  ) -> tuple[DataLoader, DataLoader, list[str]]:
    mean, std = (0.485,0.456,0.406), (0.229,0.224,0.225)

    train_tf = T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.7, 1.0), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(p=0.2),                      # only occasionally flip vertically
        T.RandomRotation(degrees=15),                     # allow slight rotation
        T.ColorJitter(0.2, 0.2, 0.2, 0.1),                 # stronger jitter
        T.RandomGrayscale(p=0.05),                        # rarely grayscale
        T.RandomPerspective(distortion_scale=0.2, p=0.3), # small perspective warping
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    val_tf = T.Compose([
        T.Resize(img_size+32, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean,std),
    ])

    # ds_train = tv.datasets.ImageFolder(root/ "train", transform=train_tf)
    ds_train = RGBImageFolder(root/ "train", transform=train_tf)

    ds_val   = tv.datasets.ImageFolder(root/ "val",   transform=val_tf)

    # ── balanced sampler to fight class imbalance ────────────────────────────
    counts = torch.tensor([ds_train.targets.count(i) for i in range(len(ds_train.classes))],
                          dtype=torch.float)
    weights = 1. / counts
    sample_weights = [weights[t] for t in ds_train.targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(ds_train, batch_size=batch, sampler=sampler,
                              num_workers=n_workers, pin_memory=True)
    val_loader   = DataLoader(ds_val,   batch_size=batch, shuffle=False,
                              num_workers=n_workers, pin_memory=True)
    return train_loader, val_loader, ds_train.classes

# ──────────────────────────────────────────────────────────────────────────────
def build_model(n_classes: int) -> torch.nn.Module:
    # torchvision 0.15+: use weights enum rather than deprecated `pretrained`
    weights = tv.models.EfficientNet_B0_Weights.DEFAULT
    model = tv.models.efficientnet_b0(weights=weights)
    # swap classifier head
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
    return model

# ──────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, pbar_desc):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc=pbar_desc, leave=False)

    for xb, yb in pbar:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(xb)
            loss = criterion(logits, yb)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        preds = model(xb).argmax(1)
        correct += (preds == yb).sum().item()
        total   += yb.size(0)
    return correct/total

# ──────────────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root   = Path(args.data)
    
    train_losses, train_accuracies = [], []
    val_accuracies = []

    
    train_loader, val_loader, class_names = build_loaders(
        root, args.img_size, args.batch, args.workers
    )
    print(f"📊 Classes: {class_names}")
    print(f"🏋️  train={len(train_loader.dataset)}   val={len(val_loader.dataset)}")

    model = build_model(len(class_names)).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # --- Scheduler ---
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)

    # --- Loss and AMP ---
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.cuda.amp.GradScaler() if args.amp and device=="cuda" else None

    best_acc = 0.0
    patience = 5
    no_improve = 0

    # --- Handle Ctrl‑C ---
    def save_and_quit(signum, frame):
        torch.save(model.state_dict(), args.checkpoint.replace(".pth", "_interrupt.pth"))
        print("\n💾 Interrupted – model saved. Bye!")
        sys.exit(0)
    signal.signal(signal.SIGINT, save_and_quit)

    # --- Training Loop ---
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, opt, scaler, device, f"[{epoch}/{args.epochs}]")
        
        val_acc = evaluate(model, val_loader, device)
        scheduler.step(epoch)  # required by WarmRestarts

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        
        dt = time.perf_counter() - t0
        print(f"Epoch {epoch:>2}/{args.epochs}  "
                f"loss {train_loss:.4f}  train-acc {train_acc:.3%}  val-acc {val_acc:.3%}  "
                f"lr {opt.param_groups[0]['lr']:.2e}  [{dt:.1f}s]")

        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), args.checkpoint)
            print(f"  🏅 New best – model saved to {args.checkpoint}")
        else:
            no_improve += 1
            print(f"⚠️  No improvement ({no_improve}/{patience})")

        if no_improve >= patience:
            print("🛑 Early stopping triggered.")
            break
    
    df = pd.DataFrame({
        'epoch': list(range(1, args.epochs+1)),
        'train_loss': train_losses,
        'train_acc': train_accuracies,
        'val_acc': val_accuracies
    })
    df.to_csv("training_metrics.csv", index=False)
    print("📈 Training metrics saved to training_metrics.csv")
    
    print(f"✅ Finished.  Best val accuracy: {best_acc:.2%}")

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()                # (Windows multi‑process dataloader fix)
    main()
