"""
For each TTA step, compute the per-step parameter delta,
serialize & compress with zlib, report raw vs compressed size.
"""
import os, zlib, torch, numpy as np, struct

SAVE_DIR = "./saved_params_ams"
DATASETS = [
    "I", "A", "R", "K", "V",
    "dtd", "flower102", "food101", "cars", "sun397",
    "aircraft", "pets", "caltech101", "ucf101", "eurosat",
]
N_SAMPLES = 5


def delta_to_bytes(delta_dict):
    """Concatenate all delta tensors (fp16) into raw bytes."""
    parts = []
    for name in sorted(delta_dict.keys()):
        parts.append(delta_dict[name].half().numpy().tobytes())
    return b"".join(parts)


def delta_to_sparse_bytes(delta_dict):
    """
    Sparse encoding: for each tensor, store (n_nonzero, indices_u32, values_f16).
    This is closer to what would actually be transmitted.
    """
    parts = []
    for name in sorted(delta_dict.keys()):
        t = delta_dict[name].half().flatten()
        nz_mask = t != 0
        indices = nz_mask.nonzero(as_tuple=False).squeeze(-1).to(torch.int32)
        values = t[nz_mask]
        n = indices.numel()
        parts.append(struct.pack("<I", n))
        if n > 0:
            parts.append(indices.numpy().tobytes())   # 4 bytes each
            parts.append(values.numpy().tobytes())    # 2 bytes each
    return b"".join(parts)


print(f"{'dataset':<12} {'step':>5} {'Δnz%':>7} "
      f"{'raw_KB':>9} {'zlib_KB':>9} "
      f"{'sparse_KB':>10} {'sp+zl_KB':>10} {'sp+zl ratio':>12}")
print("-" * 82)

summary = {}

for ds in DATASETS:
    ds_dir = os.path.join(SAVE_DIR, ds)
    if not os.path.isdir(ds_dir):
        continue

    initial = torch.load(os.path.join(ds_dir, "initial.pt"), map_location="cpu")
    prev = initial
    ds_rows = []

    for si in range(N_SAMPLES):
        fpath = os.path.join(ds_dir, f"sample_{si}.pt")
        if not os.path.exists(fpath):
            continue
        cur = torch.load(fpath, map_location="cpu")

        delta = {}
        total_numel = 0
        nonzero_numel = 0
        for name in sorted(initial.keys()):
            d = cur[name] - prev[name]
            delta[name] = d
            total_numel += d.numel()
            nonzero_numel += (d != 0).sum().item()

        nz_pct = nonzero_numel / total_numel * 100 if total_numel else 0

        # Method 1: full dense → zlib
        raw = delta_to_bytes(delta)
        raw_zlib = zlib.compress(raw, level=1)

        # Method 2: sparse encoding → zlib
        sparse = delta_to_sparse_bytes(delta)
        sparse_zlib = zlib.compress(sparse, level=1)

        raw_kb = len(raw) / 1024
        zlib_kb = len(raw_zlib) / 1024
        sp_kb = len(sparse) / 1024
        sp_zl_kb = len(sparse_zlib) / 1024
        ratio = sp_zl_kb / raw_kb if raw_kb > 0 else 0

        print(f"{ds:<12} {si:>5} {nz_pct:>6.1f}% "
              f"{raw_kb:>9.1f} {zlib_kb:>9.1f} "
              f"{sp_kb:>10.1f} {sp_zl_kb:>10.1f} {ratio:>11.2%}")
        ds_rows.append(dict(nz=nz_pct, raw=raw_kb, zlib=zlib_kb,
                            sp=sp_kb, sp_zl=sp_zl_kb))
        prev = cur

    if ds_rows:
        avg = {k: np.mean([r[k] for r in ds_rows]) for k in ds_rows[0]}
        print(f"{'  → avg':<12} {'':>5} {avg['nz']:>6.1f}% "
              f"{avg['raw']:>9.1f} {avg['zlib']:>9.1f} "
              f"{avg['sp']:>10.1f} {avg['sp_zl']:>10.1f} "
              f"{avg['sp_zl']/avg['raw']:>11.2%}")
        summary[ds] = avg
    print()

# ---- final summary ----
print("=" * 82)
print(f"{'dataset':<12} {'Δnz%':>7} "
      f"{'raw_KB':>9} {'dense+zl':>9} "
      f"{'sparse_KB':>10} {'sp+zlib':>10} {'sp+zl ratio':>12}")
print("-" * 82)
for ds in DATASETS:
    if ds not in summary:
        continue
    a = summary[ds]
    print(f"{ds:<12} {a['nz']:>6.1f}% "
          f"{a['raw']:>9.1f} {a['zlib']:>9.1f} "
          f"{a['sp']:>10.1f} {a['sp_zl']:>10.1f} "
          f"{a['sp_zl']/a['raw']:>11.2%}")
vals = list(summary.values())
g = {k: np.mean([v[k] for v in vals]) for k in vals[0]}
print("-" * 82)
print(f"{'OVERALL':<12} {g['nz']:>6.1f}% "
      f"{g['raw']:>9.1f} {g['zlib']:>9.1f} "
      f"{g['sp']:>10.1f} {g['sp_zl']:>10.1f} "
      f"{g['sp_zl']/g['raw']:>11.2%}")
