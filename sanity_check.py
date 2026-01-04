# sanity_check.py
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

from datasets import TaskCfg, build_dataloaders

# tuoi import
from adapters.PEFTclass import PEFTViT
from adapters.LoRAs import apply_LoRA  # se vuoi provarlo


def count_params(model, trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def grads_report(model, max_lines=30):
    lines = []
    for name, p in model.named_parameters():
        if p.grad is None:
            g = "None"
        else:
            g = f"{p.grad.norm().item():.3e}"
        lines.append((name, p.requires_grad, g, p.numel()))
    # ordina: trainabili prima
    lines.sort(key=lambda x: (not x[1], x[0]))
    for i, (n, req, g, num) in enumerate(lines[:max_lines]):
        print(f"{i:02d}  req_grad={req!s:5s}  grad={g:>10s}  n={num:>9d}  {n}")
    if len(lines) > max_lines:
        print(f"... ({len(lines)-max_lines} more parameters)")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # 1) Dataloaders
    task = TaskCfg(
        name="cifar100",
        num_classes=100,
        image_size=224,
        train_k=1000,
        val_k=1000,
        seed=0,
        batch_size=16,
        num_workers=2,
    )
    train_loader, val_loader, _ = build_dataloaders(task)

    x, y = next(iter(train_loader))
    print("batch x:", x.shape, x.dtype, "y:", y.shape, y.dtype)
    print("x min/max:", x.min().item(), x.max().item())

    # 2) Backbone torchvision + weights
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    backbone = vit_b_16(weights=weights)

    # 3) Wrap in your PEFTViT
    model = PEFTViT(backbone, method='lora', nb_classes=task.num_classes, attention=True, qkv = [False, True, True])

    # OPTIONAL: apply LoRA (example: attention+mlp)
    # Attenzione: nel tuo apply_LoRA correggi la chiamata LoRASelfAttention passando qkv=qkv
    # model.model = apply_LoRA(model.model, r=8, attention=True, mlp=True, qkv=[True, False, False])

    # 4) Freeze backbone, train only head (baseline)
    # Adatta ai tuoi metodi reali:
    # - se vuoi linear probing: congela tutto e lascia solo la head
    # - se vuoi adapter tuning: congela backbone e sblocca adapter+head

    # Esempio generico: congela tutto
    model.set_trainable_parameters()

    model.to(device)

    print("Total params:", count_params(model))
    print("Trainable params:", count_params(model, trainable_only=True))

    # 5) One training step
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    model.train()
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)

    optim.zero_grad(set_to_none=True)
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    optim.step()

    print("loss:", loss.item())
    grads_report(model, max_lines=40)

    # 6) Quick eval forward (no crash)
    model.eval()
    with torch.no_grad():
        x2, _ = next(iter(val_loader))
        out = model(x2.to(device))
        print("eval logits shape:", out.shape)


if __name__ == "__main__":
    main()
