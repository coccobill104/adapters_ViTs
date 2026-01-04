# data/datasets.py
from dataclasses import dataclass
from typing import Tuple, Optional, Sequence
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from torchvision.models import ViT_B_16_Weights

# -------------------------
# Config
# -------------------------
@dataclass
class TaskCfg:
    name: str
    num_classes: int
    image_size: int = 224
    train_k: int = 1000
    val_k: int = 1000
    seed: int = 0
    batch_size: int = 64
    num_workers: int = 2
    pin_memory: bool = True


# -------------------------
# Transforms from torchvision weights
# -------------------------
def get_vit_transforms(image_size: int = 224, weights=ViT_B_16_Weights.IMAGENET1K_V1):
    """
    Returns (train_tfm, eval_tfm) consistent with the chosen ViT weights.
    eval_tfm matches weights.transforms() behavior (resize->centercrop->norm).
    train_tfm uses augmentations but the same mean/std.
    """
    # mean/std for the pretrained weights
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    # Eval transform: use weights.transforms() (most faithful)
    eval_tfm = weights.transforms()

    # Train transform: augmentation + same normalization
    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_tfm, eval_tfm


# -------------------------
# Helpers
# -------------------------
def _subset_k(ds, k: Optional[int], seed: int):
    if k is None or k >= len(ds):
        return ds
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:k].tolist()
    return Subset(ds, idx)


def _make_loaders(train_ds, val_ds, test_ds, cfg: TaskCfg):
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory
    )
    return train_loader, val_loader, test_loader


def _split_indices(n: int, fractions: Sequence[float], seed: int):
    """
    fractions: e.g. (0.8, 0.1, 0.1), must sum to 1 (approximately)
    Returns list of index lists: [idx_train, idx_val, idx_test]
    """
    assert len(fractions) == 3
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_train = int(fractions[0] * n)
    n_val = int(fractions[1] * n)
    idx_train = perm[:n_train]
    idx_val = perm[n_train:n_train + n_val]
    idx_test = perm[n_train + n_val:]
    return idx_train, idx_val, idx_test


# -------------------------
# Main builder
# -------------------------
def build_dataloaders(cfg: TaskCfg, root: str = "./data") -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_tfm, eval_tfm = get_vit_transforms(image_size=cfg.image_size)

    name = cfg.name.lower()

    if name == "cifar100":
        train_full = datasets.CIFAR100(root, train=True, download=True, transform=train_tfm)
        test_ds = datasets.CIFAR100(root, train=False, download=True, transform=eval_tfm)

        val_size = min(cfg.val_k, max(1, len(train_full) // 10))
        train_size = len(train_full) - val_size

        g = torch.Generator().manual_seed(cfg.seed)
        train_subset, val_subset = random_split(train_full, [train_size, val_size], generator=g)

        train_subset = _subset_k(train_subset, cfg.train_k, cfg.seed)

        # Recreate a second CIFAR100 train dataset with eval transform, and map val indices onto it
        train_full_eval = datasets.CIFAR100(root, train=True, download=True, transform=eval_tfm)
        # random_split gives Subset with indices referring to train_full; reuse those indices
        val_indices = val_subset.indices if hasattr(val_subset, "indices") else list(range(len(val_subset)))
        val_subset = Subset(train_full_eval, val_indices)
        val_subset = _subset_k(val_subset, cfg.val_k, cfg.seed + 1)

        return _make_loaders(train_subset, val_subset, test_ds, cfg)

    elif name == "eurosat":
        # EuroSAT comes as a single dataset => we create index splits.
        # To avoid transform leak, build separate dataset objects:
        base_no_tfm = datasets.EuroSAT(root, download=True, transform=None)
        idx_train, idx_val, idx_test = _split_indices(len(base_no_tfm), (0.8, 0.1, 0.1), cfg.seed)

        ds_train = datasets.EuroSAT(root, download=False, transform=train_tfm)
        ds_eval = datasets.EuroSAT(root, download=False, transform=eval_tfm)

        train_subset = Subset(ds_train, idx_train)
        val_subset = Subset(ds_eval, idx_val)
        test_subset = Subset(ds_eval, idx_test)

        train_subset = _subset_k(train_subset, cfg.train_k, cfg.seed)
        val_subset = _subset_k(val_subset, cfg.val_k, cfg.seed + 1)

        return _make_loaders(train_subset, val_subset, test_subset, cfg)

    elif name == "dtd":
        train_ds = datasets.DTD(root, split="train", download=True, transform=train_tfm)
        val_ds = datasets.DTD(root, split="val", download=True, transform=eval_tfm)
        test_ds = datasets.DTD(root, split="test", download=True, transform=eval_tfm)

        train_ds = _subset_k(train_ds, cfg.train_k, cfg.seed)
        val_ds = _subset_k(val_ds, cfg.val_k, cfg.seed + 1)

    elif name == "flowers102":
        train_full = datasets.Flowers102(root, split="train", download=True, transform=train_tfm)
        val_ds = datasets.Flowers102(root, split="val", download=True, transform=eval_tfm)
        test_ds = datasets.Flowers102(root, split="test", download=True, transform=eval_tfm)
        
        train_ds = _subset_k(train_full, cfg.train_k, cfg.seed)
        val_ds = _subset_k(val_ds, cfg.val_k, cfg.seed+1)


    elif name == "svhn":
        train_full = datasets.SVHN(root, split="train", download=True, transform=train_tfm)
        test_ds = datasets.SVHN(root, split="test", download=True, transform=eval_tfm)

        val_size = min(cfg.val_k, max(1, len(train_full)//10))
        train_size = len(train_full) - val_size
        g = torch.Generator().manual_seed(cfg.seed)
        train_subset, val_subset = random_split(train_full, [train_size, val_size], generator=g)

        train_subset = _subset_k(train_subset, cfg.train_k, cfg.seed)

        # Recreate SVHN train dataset with eval transform for val
        train_full_eval = datasets.SVHN(root, split="train", download=False, transform=eval_tfm)
        val_indices = val_subset.indices if hasattr(val_subset, "indices") else list(range(len(val_subset)))
        val_subset = Subset(train_full_eval, val_indices)
        val_subset = _subset_k(val_subset, cfg.val_k, cfg.seed + 1)

        train_ds, val_ds = train_subset, val_subset

    else:
        raise ValueError(f"Unknown dataset name: {cfg.name}. Supported: cifar100, eurosat, dtd, flowers102, svhn")
    

    return _make_loaders(train_ds, val_ds, test_ds, cfg)
