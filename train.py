import argparse, yaml, time
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from datasets import TaskCfg, build_dataloaders
from models.build import build_model
from utils.seed import set_seed
from utils.metrics import accuracy_top1
from utils.logging import append_result, Timer

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    accs = []
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        logits = model(x)
        accs.append(accuracy_top1(logits, y))
    return sum(accs)/len(accs)

def train_one(cfg):
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    task = TaskCfg(**cfg["task"])
    train_loader, val_loader, test_loader = build_dataloaders(task, root=cfg.get("data_root","./data"))

    model = build_model(cfg["task"]["num_classes"], cfg["method"], cfg.get("adapter",{}))
    model.to(device)

    model.set_trainable_parameters()
    trainable = count_trainable_params(model)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"].get("weight_decay", 0.0),
    )
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and cfg.get("amp", True)
    scaler = GradScaler(enabled=use_amp)

    best_val = -1
    best_state = None
    patience = cfg.get("patience", 5)
    bad = 0

    for epoch in range(cfg["epochs"]):
        model.train()
        timer = Timer(); timer.start()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        for x,y in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            x,y = x.to(device), y.to(device)
            optim.zero_grad(set_to_none=True)
            #with autocast(enabled=(device=="cuda" and cfg.get("amp", True))):
            with autocast(device_type="cuda", enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()


        epoch_time = timer.stop()
        peak_mem = (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else 0.0

        val_acc = evaluate(model, val_loader, device)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k:v.detach().cpu() for k,v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)

    test_acc = evaluate(model, test_loader, device)

    row = {
        "task": cfg["task"]["name"],
        "method": cfg["method"],
        "seed": cfg["seed"],
        "val_acc": best_val,
        "test_acc": test_acc,
        "trainable_params": trainable,
        "peak_mem_gb": peak_mem,
        "epochs_ran": epoch+1,
        "backbone": cfg["backbone"],
    }
    append_result(cfg.get("results_csv","./results/results.csv"), row)
    print(row)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    train_one(cfg)
