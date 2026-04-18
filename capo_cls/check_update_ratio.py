"""Check that the fraction of actually-updated text_encoder params is ~10% (gamma=0.1)."""
import os, torch

SAVE_DIR = "./saved_params_ams"
DATASETS = [
    "I", "A", "R", "K", "V",
    "dtd", "flower102", "food101", "cars", "sun397",
    "aircraft", "pets", "caltech101", "ucf101", "eurosat",
]

print(f"{'dataset':<12} {'sample':>7} {'total':>12} {'changed':>10} {'ratio':>8}")
print("-" * 54)

for ds in DATASETS:
    ds_dir = os.path.join(SAVE_DIR, ds)
    if not os.path.isdir(ds_dir):
        print(f"{ds:<12}  ** missing **")
        continue
    initial = torch.load(os.path.join(ds_dir, "initial.pt"), map_location="cpu")
    for si in range(5):
        fpath = os.path.join(ds_dir, f"sample_{si}.pt")
        if not os.path.exists(fpath):
            continue
        updated = torch.load(fpath, map_location="cpu")
        total_params = 0
        changed_params = 0
        for name in initial:
            init_v = initial[name]
            upd_v = updated[name]
            total_params += init_v.numel()
            changed_params += (init_v != upd_v).sum().item()
        ratio = changed_params / total_params if total_params > 0 else 0
        print(f"{ds:<12} {si:>7} {total_params:>12,} {changed_params:>10,} {ratio:>8.2%}")
