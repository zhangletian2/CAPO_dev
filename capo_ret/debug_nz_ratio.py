"""Debug: check per-param non-zero ratio to explain why T2I ≈ 6% instead of 10%."""
import os, torch

SAVE_DIR = "./saved_params_ams_ret"

for task in ["t2i", "i2t"]:
    task_dir = os.path.join(SAVE_DIR, task)
    initial = torch.load(os.path.join(task_dir, "initial.pt"), map_location="cpu")
    sample = torch.load(os.path.join(task_dir, "sample_2.pt"), map_location="cpu")

    print(f"\n=== {task.upper()} (sample_2 vs initial) ===")
    print(f"{'param_name':<50} {'numel':>10} {'Δnz':>10} {'ratio':>8}")
    print("-" * 82)

    total, total_nz = 0, 0
    for name in sorted(initial.keys()):
        d = sample[name] - initial[name]
        nz = (d != 0).sum().item()
        n = d.numel()
        total += n
        total_nz += nz
        print(f"{name:<50} {n:>10,} {nz:>10,} {nz/n*100:>7.2f}%")

    print("-" * 82)
    print(f"{'TOTAL':<50} {total:>10,} {total_nz:>10,} {total_nz/total*100:>7.2f}%")
