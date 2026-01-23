# train_imagenet.py
import os
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ----------------------------
# Import / paste your model here
# from model import preact_resnet18_bottleneck
# ----------------------------

# Example: assume preact_resnet18_bottleneck is already defined in this file.


def accuracy_topk(logits, targets, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)  # (B,maxk)
        pred = pred.t()  # (maxk,B)
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            res.append(correct[:k].reshape(-1).float().sum().mul_(100.0 / targets.size(0)))
        return res


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, log_every=50):
    model.train()
    running_loss = 0.0
    running_top1 = 0.0
    running_top5 = 0.0
    n = 0

    t0 = time.time()
    for it, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        top1, top5 = accuracy_topk(logits, targets, topk=(1, 5))

        bs = images.size(0)
        running_loss += loss.item() * bs
        running_top1 += top1.item() * bs
        running_top5 += top5.item() * bs
        n += bs

        if (it + 1) % log_every == 0:
            dt = time.time() - t0
            print(
                f"  iter {it+1:5d}/{len(loader)} | "
                f"loss {running_loss/n:.4f} | top1 {running_top1/n:.2f}% | top5 {running_top5/n:.2f}% | "
                f"{dt:.1f}s"
            )

    return running_loss / n, running_top1 / n, running_top5 / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_top1 = 0.0
    running_top5 = 0.0
    n = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)
        top1, top5 = accuracy_topk(logits, targets, topk=(1, 5))

        bs = images.size(0)
        running_loss += loss.item() * bs
        running_top1 += top1.item() * bs
        running_top5 += top5.item() * bs
        n += bs

    return running_loss / n, running_top1 / n, running_top5 / n


def save_checkpoint(path, model, optimizer, epoch, best_top1):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_top1": best_top1,
    }
    torch.save(ckpt, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True,
                        help="Path to ImageNet folder with train/ and val/ subfolders")
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # --------- Data ---------
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    train_dir = os.path.join(args.data, "train")
    val_dir = os.path.join(args.data, "val")

    train_set = datasets.ImageFolder(train_dir, transform=train_tf)
    val_set = datasets.ImageFolder(val_dir, transform=val_tf)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    print(f"Train samples: {len(train_set)} | Val samples: {len(val_set)}")

    # --------- Model ---------
    model = preact_resnet18_bottleneck(num_classes=1000, in_ch=3)
    model = model.to(device)

    # --------- Loss / Optim / Scheduler ---------
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)

    # Classic ImageNet schedule: step LR at 30/60/80
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.1)

    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None

    # --------- Resume ---------
    start_epoch = 0
    best_top1 = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_top1 = ckpt.get("best_top1", 0.0)
        print(f"Resumed from {args.resume} @ epoch {start_epoch}, best_top1={best_top1:.2f}")

    # --------- Train ---------
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs} | lr={optimizer.param_groups[0]['lr']:.5f}")

        train_loss, train_top1, train_top5 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler=scaler, log_every=50
        )
        val_loss, val_top1, val_top5 = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1:3d} | "
            f"train: loss {train_loss:.4f} top1 {train_top1:.2f}% top5 {train_top5:.2f}% | "
            f"val: loss {val_loss:.4f} top1 {val_top1:.2f}% top5 {val_top5:.2f}%"
        )

        scheduler.step()

        # save best
        if val_top1 > best_top1:
            best_top1 = val_top1
            save_checkpoint(save_dir / "best.pt", model, optimizer, epoch, best_top1)
            print(f"  saved best.pt (top1={best_top1:.2f}%)")

        # save last
        save_checkpoint(save_dir / "last.pt", model, optimizer, epoch, best_top1)

    print("\nDone. Best top1:", best_top1)


if __name__ == "__main__":
    main()