import torch

@torch.no_grad()
def accuracy_top1(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()
