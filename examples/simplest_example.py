# train_stl10_256.py
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader,random_split
from torchvision import datasets, transforms
import sys,os

sys.path.insert(0,os.getcwd()) 
if "../" not in sys.path:
    sys.path.insert(0,"../")
print(sys.path,os.getcwd())

# ----------------------------
# Paste / import your model here
# from model import preact_resnet18_bottleneck
# ----------------------------
from modules.lowrank import preact_resnet18_bottleneck
from utils.imagenet import ImageNetCustom

def accuracy_topk(logits, targets, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        out = []
        for k in topk:
            out.append(correct[:k].reshape(-1).float().sum().mul_(100.0 / targets.size(0)))
        return out


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, log_every=50):
    model.train()
    total_loss, total_top1, total_top5, n = 0.0, 0.0, 0.0, 0
    for it, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        #print(images)
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
        total_loss += loss.item() * bs
        total_top1 += top1.item() * bs
        total_top5 += top5.item() * bs
        n += bs

        if (it + 1) % log_every == 0:
            print(
                f"iter {it+1:5d}/{len(loader)} | "
                f"loss {total_loss/n:.4f} | top1 {total_top1/n:.2f}% | top5 {total_top5/n:.2f}%"
            )

    return total_loss / n, total_top1 / n, total_top5 / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_top1, total_top5, n = 0.0, 0.0, 0.0, 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)
        top1, top5 = accuracy_topk(logits, targets, topk=(1, 5))

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_top1 += top1.item() * bs
        total_top5 += top5.item() * bs
        n += bs

    return total_loss / n, total_top1 / n, total_top5 / n


def save_checkpoint(path, model, optimizer, epoch, best_top1):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_top1": best_top1,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data_stl10")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save-dir", type=str, default="./checkpoints_stl10_256")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    args, _ = parser.parse_known_args()  # colab-friendly

    torch.manual_seed(args.seed)
    cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ---- Transforms (force 256x256) ----
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(512, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize(mean=(0.485, 0.456, 0.406),
        #                     std=(0.229, 0.224, 0.225)),
    ])
    val_tf = transforms.Compose([
        #transforms.Resize(288),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        #transforms.Normalize(mean=(0.485, 0.456, 0.406),
        #                     std=(0.229, 0.224, 0.225)),
    ])
    #"""
    # ---- Caltech-256 ----
    """full_set = datasets.Caltech256(
        root=args.data,
        download= True, #True,
        transform=train_tf  # temp, we'll override val below
    )
    """
    full_set = ImageNetCustom(
        root="/home/quinoa/imagenet",
        split="train",
        transform=train_tf  # temp, we'll override val below
    )
    #print(full_set.index,"\n", full_set.y)

    #print(full_set)

    num_classes = 1000
    total = len(full_set)

    # 90/10 train/val split
    #train_size = int(0.4 * total)
    #val_size = int(0.2*(total - train_size))
    train_size = int(0.8*total)
    val_size = int(0.008*total)
    #val_size = int(0.015*(total - train_size))
    #val_size = total - train_size
    trash_size = total - train_size - val_size

    train_set, val_set,_ = random_split(full_set, [train_size, val_size,trash_size])
    #train_set, val_set = random_split(full_set, [train_size, val_size])
    #print(total,train_size,val_size,trash_size)
    # Apply val transform to val subset
    val_set.dataset.transform = val_tf


    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        #collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        #collate_fn = collate_fn
    )
    #"""

    # ---- STL10 ----
    """
    train_set = datasets.STL10(root=args.data, split="train", download=True, transform=train_tf)
    val_set = datasets.STL10(root=args.data, split="test", download=True, transform=val_tf)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    num_classes = 10


    # ---- GTSRB ----
    train_set = datasets.GTSRB(
        root=args.data,
        split="train",
        download=True,
        transform=train_tf
    )

    val_set = datasets.GTSRB(
        root=args.data,
        split="test",
        download=True,
        transform=val_tf
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    num_classes = 43
    """

    print(f"Train samples: {len(train_set)} | Test samples: {len(val_set)} | Classes: {num_classes}")


    # ---- Model ----
    model = preact_resnet18_bottleneck(num_classes=num_classes, in_ch=3,useLR=True).to(device)
    ## Debugging the model
    #print(model)

    # ---- Loss / Optim / Scheduler ----
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=[0.9,0.99], weight_decay=args.wd,eps=6e-2)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,100,160,180], gamma=0.1)

    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None

    # ---- Resume ----
    start_epoch = 0
    best_top1 = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_top1 = ckpt.get("best_top1", 0.0)
        print(f"Resumed from {args.resume} @ epoch {start_epoch}, best_top1={best_top1:.2f}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- Train ----
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs} | lr={optimizer.param_groups[0]['lr']:.5f}")

        tr_loss, tr_top1, tr_top5 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler=scaler, log_every=50
        )
        va_loss, va_top1, va_top5 = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1:3d} | "
            f"train: loss {tr_loss:.4f} top1 {tr_top1:.2f}% top5 {tr_top5:.2f}% | "
            f"test:  loss {va_loss:.4f} top1 {va_top1:.2f}% top5 {va_top5:.2f}%"
        )

        scheduler.step()

        save_checkpoint(save_dir / "last.pt", model, optimizer, epoch, best_top1)
        if va_top1 > best_top1:
            best_top1 = va_top1
            save_checkpoint(save_dir / "best.pt", model, optimizer, epoch, best_top1)
            print(f"  saved best.pt (top1={best_top1:.2f}%)")

    print("\nDone. Best top1:", best_top1)

import sys

sys.argv = [
    "train_stl10_256.py",
    "--data", "../curated",
    "--epochs", "400",
    "--batch-size", "80",
    "--lr", "0.08",
    "--amp",
    "--label-smoothing", "0.15",
]

main()


if __name__ == "__main__":
    main()
