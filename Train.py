import os, argparse
import torch, torch.nn as nn
from torch.optim import SGD, Adam, Adagrad, RMSprop, NAdam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import wandb

from Model import VGG6
from DataModel import get_dataloaders

# --- allow --autoaugment to accept True/False from wandb sweeps ---
def str2bool(x):
    if isinstance(x, bool):
        return x
    return str(x).lower() in ("1", "true", "t", "y", "yes")

def set_seed(seed=42):
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def accuracy_top1(logits, targets):
    with torch.no_grad():
        return (logits.argmax(dim=1) == targets).float().mean().item()

@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval(); tot_loss = tot_acc = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        b = y.size(0)
        tot_loss += loss.item() * b
        tot_acc  += accuracy_top1(logits, y) * b
        n += b
    return tot_loss / n, tot_acc / n

def build_optimizer(params, cfg):
    opt = cfg.optimizer.lower()
    if opt == "sgd":          return SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    if opt == "sgd_nesterov": return SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay, nesterov=True)
    if opt == "adam":         return Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if opt == "adamw":        return AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if opt == "adagrad":      return Adagrad(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if opt == "rmsprop":      return RMSprop(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    if opt == "nadam":        return NAdam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

def build_scheduler(optimizer, cfg):
    s = cfg.scheduler.lower()
    if s == "cosine":  return CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    if s == "step":    return StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    if s == "plateau": return ReduceLROnPlateau(optimizer, mode="max", patience=10, factor=0.5)
    if s == "none":    return None
    raise ValueError(f"Unknown scheduler: {cfg.scheduler}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="cs6886-vgg6-cifar10")
    p.add_argument("--entity",  default=None)

    # Q2 knobs
    p.add_argument("--activation", default="relu", choices=["relu","silu","gelu","tanh","sigmoid"])
    p.add_argument("--optimizer",  default="sgd",  choices=["sgd","sgd_nesterov","adam","adagrad","rmsprop","nadam"])
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--scheduler", default="cosine", choices=["cosine","step","plateau","none"])
    p.add_argument("--step_size", type=int, default=30)
    p.add_argument("--gamma", type=float, default=0.1)
    # NEW: boolean-friendly autoaugment (accepts --autoaugment, --autoaugment=True/False)
    p.add_argument("--autoaugment", type=str2bool, nargs="?", const=True, default=False)
    p.add_argument("--label_smoothing", type=float, default=0.0)

    # infra / reproducibility
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--save_dir", default="checkpoints")
    args = p.parse_args()

    # W&B init (sweep can override via config)
    run = wandb.init(project=args.project, entity=args.entity, config=vars(args))
    cfg = argparse.Namespace(**dict(wandb.config))

    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=cfg.batch_size, num_workers=cfg.num_workers,
        seed=cfg.seed, use_autoaugment=cfg.autoaugment
    )

    # Model / opt / sched
    model = VGG6(activation=cfg.activation, dropout=0.3, batch_norm=True).to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    opt = build_optimizer(model.parameters(), cfg)
    sch = build_scheduler(opt, cfg)

    wandb.watch(model, log="all", log_freq=100)

    # W&B scatter fix: collect and log once at the end (no immutable-table mutation)
    val_points = []  # list of (global_step, val_acc)

    scaler = GradScaler()
    best_val, global_step = 0.0, 0
    os.makedirs(cfg.save_dir, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for x, y in tqdm(train_loader, leave=False, desc=f"Epoch {epoch}/{cfg.epochs}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with autocast():
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            global_step += 1

        tr_loss, tr_acc = evaluate(model, train_loader, loss_fn, device)
        va_loss, va_acc = evaluate(model, val_loader,  loss_fn, device)

        if isinstance(sch, ReduceLROnPlateau): sch.step(va_acc)
        elif sch is not None:                  sch.step()

        wandb.log({
            "epoch": epoch,
            "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": va_loss,   "val_acc": va_acc,
            "lr": opt.param_groups[0]["lr"]
        }, step=global_step)

        # collect for scatter
        val_points.append((global_step, va_acc))

        # save best
        if va_acc > best_val:
            best_val = va_acc
            path = os.path.join(cfg.save_dir, "best.pt")
            torch.save({"model_state": model.state_dict(),
                        "cfg": dict(wandb.config),
                        "val_acc": best_val}, path)
            wandb.save(path)

    # final test
    te_loss, te_acc = evaluate(model, test_loader, loss_fn, device)
    wandb.log({"test_loss": te_loss, "test_acc": te_acc})
    print(f"[FINAL] Test accuracy: {te_acc:.4f}")

    # single scatter at the end (no table mutation)
    table = wandb.Table(data=val_points, columns=["step", "val_acc"])
    wandb.log({
        "val_acc_vs_step": wandb.plot.scatter(
            table, "step", "val_acc", title="Validation Accuracy vs Step"
        )
    })

    run.finish()

if __name__ == "__main__":
    main()
