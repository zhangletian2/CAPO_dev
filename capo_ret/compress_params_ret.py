"""
Per-step delta compression stats for AMS retrieval (T2I / I2T).

Only includes parameters that actually participate in gradient updates.
Excludes: token_embedding.weight (sparse grad, ~0% effective update),
          logit_scale (scalar, no gradient in TTA forward).
"""
import os, zlib, struct, torch, numpy as np

SAVE_DIR = "./saved_params_ams_ret"
TASKS = ["t2i", "i2t"]
N_SAMPLES = 5

SKIP_PARAMS = {"token_embedding.weight", "logit_scale"}


def _filter(state):
    return {k: v for k, v in state.items()
            if not any(sk in k for sk in SKIP_PARAMS)}


def delta_to_bytes(delta_dict):
    parts = []
    for name in sorted(delta_dict.keys()):
        parts.append(delta_dict[name].half().numpy().tobytes())
    return b"".join(parts)


def delta_to_sparse_bytes(delta_dict):
    parts = []
    for name in sorted(delta_dict.keys()):
        t = delta_dict[name].half().flatten()
        nz_mask = t != 0
        indices = nz_mask.nonzero(as_tuple=False).squeeze(-1).to(torch.int32)
        values = t[nz_mask]
        n = indices.numel()
        parts.append(struct.pack("<I", n))
        if n > 0:
            parts.append(indices.numpy().tobytes())
            parts.append(values.numpy().tobytes())
    return b"".join(parts)


print(f"{'task':<6} {'step':>5} {'Δnz%':>7} "
      f"{'raw_KB':>9} {'zlib_KB':>9} "
      f"{'sparse_KB':>10} {'sp+zl_KB':>10} {'sp+zl ratio':>12}")
print("-" * 76)

summary = {}

for task in TASKS:
    task_dir = os.path.join(SAVE_DIR, task)
    if not os.path.isdir(task_dir):
        print(f"{task:<6}  ** missing **")
        continue

    initial_raw = torch.load(os.path.join(task_dir, "initial.pt"),
                             map_location="cpu")
    initial = _filter(initial_raw)
    n_total = sum(v.numel() for v in initial.values())
    n_skipped = sum(v.numel() for v in initial_raw.values()) - n_total
    if n_skipped:
        print(f"  [{task}] skipped {n_skipped:,} params "
              f"(token_embedding/logit_scale), kept {n_total:,}")
    rows = []

    for si in range(N_SAMPLES):
        fpath = os.path.join(task_dir, f"sample_{si}.pt")
        if not os.path.exists(fpath):
            continue
        cur = _filter(torch.load(fpath, map_location="cpu"))

        delta = {}
        total_numel = 0
        nonzero_numel = 0
        for name in sorted(initial.keys()):
            d = cur[name] - initial[name]
            delta[name] = d
            total_numel += d.numel()
            nonzero_numel += (d != 0).sum().item()

        nz_pct = nonzero_numel / total_numel * 100 if total_numel else 0

        if nonzero_numel == 0:
            print(f"{task:<6} {si:>5}   0.0%   ** skipped (GradScaler overflow, no update) **")
            continue

        raw = delta_to_bytes(delta)
        raw_zlib = zlib.compress(raw, level=1)
        sparse = delta_to_sparse_bytes(delta)
        sparse_zlib = zlib.compress(sparse, level=1)

        raw_kb = len(raw) / 1024
        zlib_kb = len(raw_zlib) / 1024
        sp_kb = len(sparse) / 1024
        sp_zl_kb = len(sparse_zlib) / 1024
        ratio = sp_zl_kb / raw_kb if raw_kb > 0 else 0

        print(f"{task:<6} {si:>5} {nz_pct:>6.1f}% "
              f"{raw_kb:>9.1f} {zlib_kb:>9.1f} "
              f"{sp_kb:>10.1f} {sp_zl_kb:>10.1f} {ratio:>11.2%}")
        rows.append(dict(nz=nz_pct, raw=raw_kb, zlib=zlib_kb,
                         sp=sp_kb, sp_zl=sp_zl_kb))

    if rows:
        avg = {k: np.mean([r[k] for r in rows]) for k in rows[0]}
        print(f"{'  avg':<6} {'':>5} {avg['nz']:>6.1f}% "
              f"{avg['raw']:>9.1f} {avg['zlib']:>9.1f} "
              f"{avg['sp']:>10.1f} {avg['sp_zl']:>10.1f} "
              f"{avg['sp_zl']/avg['raw']:>11.2%}")
        summary[task] = avg
    print()

print("=" * 76)
print(f"{'task':<6} {'Δnz%':>7} {'raw_KB':>9} {'dense+zl':>9} "
      f"{'sparse_KB':>10} {'sp+zlib':>10} {'sp+zl ratio':>12}")
print("-" * 76)
for task in TASKS:
    if task not in summary:
        continue
    a = summary[task]
    print(f"{task:<6} {a['nz']:>6.1f}% {a['raw']:>9.1f} {a['zlib']:>9.1f} "
          f"{a['sp']:>10.1f} {a['sp_zl']:>10.1f} "
          f"{a['sp_zl']/a['raw']:>11.2%}")
