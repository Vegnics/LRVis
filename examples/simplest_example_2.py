# train_stl10_256.py
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader,random_split
from torchvision import datasets, transforms
import sys,os

from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.feature_extraction import get_graph_node_names


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

class PadToSquare:
    def __call__(self, img):
        w, h = img.size
        max_dim = max(w, h)
        pad_w = max_dim - w
        pad_h = max_dim - h
        padding = (
            pad_w // 2,
            pad_h // 2,
            pad_w - pad_w // 2,
            pad_h - pad_h // 2,
        )
        return F.pad(img, padding, fill=0)

def printLog(message,fname):
    with open(fname,"a") as file:
        file.write("\n"+message)
    print(message)

def newLog(fname):
    with open(fname,"w") as file:
        file.write("")
    print(f"{fname} created succesfully!!")

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

import torch
from torch.nn.utils import clip_grad_norm_


LOGNAME = "train_logger.txt"
newLog(LOGNAME)

def train_one_epoch(
    model,tmodel, loader,val_loader, criterion,criterion_dist, optimizer, device,
    scaler=None, log_every=50,
    accum_steps=1, grad_clip=None
):
    model.train()
    total_loss, total_top1, total_top5, n = 0.0, 0.0, 0.0, 0
    tlossce = 0.0
    tlosskd = 0.0
    
    optimizer.zero_grad(set_to_none=True)
    f_extr = create_feature_extractor(model, ["layer1.1.interpolate","layer2.1.interpolate","layer3.1.interpolate","layer4.1.interpolate","fc"])
    f_extr_t = create_feature_extractor(tmodel, ["layer1.1.add","layer2.1.add","layer3.1.add","layer4.1.add"])
    wce = 1.0
    wkd = 0.6
    for it, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        images_t = F.interpolate(images, size=(224,224), mode='bilinear', align_corners=False)
        images_t = images_t.to(torch.float32)
        targets = targets.to(device, non_blocking=True)
        if scaler is not None:
            with torch.amp.autocast("cuda",dtype=torch.float16):
                feats = list(f_extr(images).values())
                logits = feats[4] #model(images)
                loss_ce = criterion(logits, targets)
            loss_dist = 0.0
            #feats = list(f_extr(images).values())
            with torch.no_grad():
                featst = list(f_extr_t(images_t).values())
            for l in range(4):
                vals = feats[l]
                vals_t = featst[l]
                loss_dist += criterion_dist(vals.to(torch.float32),vals_t.to(torch.float32))
            loss_dist/=4
            loss = (wce * loss_ce + wkd * loss_dist)/accum_steps
            scaler.scale(loss).backward()

            # step every accum_steps
            if (it + 1) % accum_steps == 0:
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), grad_clip)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            logits = model(images)
            loss_ce = criterion(logits, targets) #/ accum_steps
            loss_dist = 0.0
            for l in range(len(layernames)):
                vals = f_extr(model).values()[l]
                vals_t = f_extr_t(tmodel).values()[l]
                loss_dist += criterion_dist(vals,vals_t)
            loss = (loss_ce+0.3*loss_dist)/accum_steps            
            loss.backward()

            if (it + 1) % accum_steps == 0:
                if grad_clip is not None:
                    clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        # logging metrics should use the *un-divided* loss
        with torch.no_grad():
            top1, top5 = accuracy_topk(logits, targets, topk=(1, 5))

        bs = images.size(0)
        tlossce += loss_ce.item() * bs
        tlosskd += loss_dist.item() * bs
        total_loss += ((wce * loss_ce.item() + wkd * loss_dist.item()) * bs)
        total_top1 += top1.item() * bs
        total_top5 += top5.item() * bs
        n += bs

        if (it + 1) % log_every == 0:
            printLog(
                f"iter {it+1:5d}/{len(loader)} | "
                f"loss {total_loss/n:.4f} | top1 {total_top1/n:.2f}% | top5 {total_top5/n:.2f}% |"
                f"Loss distill: {tlosskd/n:.4f}, Loss CE: {tlossce/n:.4f}",
                LOGNAME
            )
            #print(f"iter {it+1:5d}/{len(loader)}| Loss distill: {tlossce/n:.4f}, Loss CE: {tlosskd/n:.4f}")
        if ((it + 1) % (log_every*6)) == 0:
            va_loss, va_top1, va_top5 = validate(model, val_loader, criterion, device,max_cnt=10)
            printLog(
                f"[Validation] Iteration {it+1:3d} | "
                #f"train: loss {total_loss/n:.4f} top1 {total_top1/n:.2f}% top5 {total_top5/n:.2f}% | "
                f"test:  loss {va_loss:.4f} top1 {va_top1:.2f}% top5 {va_top5:.2f}%",
                LOGNAME
            )
            model.train()

    # if loader size not divisible by accum_steps, flush the remainder
    remainder = len(loader) % accum_steps
    if remainder != 0:
        if scaler is not None:
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            if grad_clip is not None:
                clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / n, total_top1 / n, total_top5 / n

@torch.no_grad()
def validate(model, loader, criterion, device,max_cnt=None):
    model.eval()
    total_loss, total_top1, total_top5, n = 0.0, 0.0, 0.0, 0
    cnt = 0 #if cnt is not None else None
    for k,(images, targets) in enumerate(loader):
        if max_cnt is not None and (k%3)!=0:
            continue
        if max_cnt is not None and cnt>max_cnt:
            print("[Validation]: Broken")
            break
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
        if max_cnt is not None:
            cnt+=1

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
        #PadToSquare(),
        transforms.RandomResizedCrop(256, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    val_tf = transforms.Compose([
        #transforms.Resize(288),
        #PadToSquare(),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    train_set = ImageNetCustom(
        root="/home/quinoa/imagenet",
        split="train",
        transform=train_tf  # temp, we'll override val below
    )
    
    val_set = ImageNetCustom(
        root="/home/quinoa/imagenet",
        split="val",
        transform=val_tf  # temp, we'll override val below
    )

    num_classes = 1000

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
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        #collate_fn = collate_fn
    )

    print(f"Train samples: {len(train_set)} | Test samples: {len(val_set)} | Classes: {num_classes}")


    # ---- Model ----
    model = preact_resnet18_bottleneck(num_classes=num_classes, in_ch=3,nblocks=2,useLR=True).to(device)
    # Load pretrained model
    model_teach = resnet18(weights="IMAGENET1K_V1")
    model_teach.eval()
    train_nodes, eval_nodes = get_graph_node_names(model)
    print(eval_nodes)
    for p in model_teach.parameters():
        p.requires_grad_(False)
    model_teach.to(device)

    # ---- Loss / Optim / Scheduler ----
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)
    critdistill = nn.MSELoss().to(device)
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=[0.9,0.99], weight_decay=args.wd,eps=1e-8)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,40,60,80,90], gamma=0.1)

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
        printLog(f"\nEpoch {epoch+1}/{args.epochs} | lr={optimizer.param_groups[0]['lr']:.5f}",LOGNAME)

        tr_loss, tr_top1, tr_top5 = train_one_epoch(
            model,model_teach, train_loader,val_loader, criterion,critdistill, optimizer, device, scaler=scaler, log_every=50, accum_steps=4, grad_clip=None)
        va_loss, va_top1, va_top5 = validate(model, val_loader, criterion, device)

        printLog(
            f"Epoch {epoch+1:3d} | "
            f"train: loss {tr_loss:.4f} top1 {tr_top1:.2f}% top5 {tr_top5:.2f}% | "
            f"test:  loss {va_loss:.4f} top1 {va_top1:.2f}% top5 {va_top5:.2f}%",
            LOGNAME
        )

        scheduler.step()

        save_checkpoint(save_dir / "last.pt", model, optimizer, epoch, best_top1)
        if va_top1 > best_top1:
            best_top1 = va_top1
            save_checkpoint(save_dir / "best.pt", model, optimizer, epoch, best_top1)
            printLog(f"  saved best.pt (top1={best_top1:.2f}%)",LOGNAME)

    print("\nDone. Best top1:", best_top1)

import sys

sys.argv = [
    "train_stl10_256.py",
    "--data", "../curated",
    "--epochs", "100",
    "--batch-size", "256",
    "--lr", "0.01",
    "--amp",
    "--label-smoothing", "0.15",
]

main()


if __name__ == "__main__":
    main()
