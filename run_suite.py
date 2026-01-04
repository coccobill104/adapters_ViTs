import itertools, os, subprocess, yaml

TASKS = [
    {"name":"cifar100", "num_classes":100},
    {"name":"eurosat", "num_classes":10},
    {"name":"dtd", "num_classes":47},
]
METHODS = ["linear", "lora", "vera", "ia3"]
SEEDS = [0]
BASE = {
    "backbone": "vit_base_patch16_224",
    "epochs": 30,
    "patience": 5,
    "amp": True,
    "optim": {"lr": 3e-4, "weight_decay": 0.0},
    "task": {"image_size":224, "train_k":1000, "val_k":1000, "batch_size":64, "num_workers":2},
    "adapter": {"r":8, "alpha":16, "dropout":0.0},  # usato solo da LoRA/VeRA/IA3 se serve
    "results_csv": "./results/results.csv",
}

os.makedirs("./tmp_cfgs", exist_ok=True)

for task, method, seed in itertools.product(TASKS, METHODS, SEEDS):
    cfg = dict(BASE)
    cfg["method"] = method
    cfg["seed"] = seed
    cfg["task"] = dict(BASE["task"])
    cfg["task"].update(task)

    path = f"./tmp_cfgs/{task['name']}_{method}_s{seed}.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)

    subprocess.run(["python", "train.py", "--config", path], check=False)
