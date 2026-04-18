"""
Fast neural image codec: rANS entropy coding replaces arithmetic coding subprocess.
Key optimizations:
  1. Pre-computed CDF cache (one-time ~20ms, zero per-image cost)
  2. Sigma-only shared CDFs + per-symbol mu pre-shifting
  3. Adaptive CDF window (small for qp1-3, large for qp4-7)
  4. Parallel z1/z2/z3 encoding via persistent ThreadPool
  5. No subprocess (direct pybind11 C++ rANS)
"""

import math
import struct
import time
import numpy as np
import torch
from PIL import Image
from scipy.special import erf as scipy_erf
from compressai.ans import np_encode_with_indexes, np_decode_with_indexes
from concurrent.futures import ThreadPoolExecutor

PRECISION_BITS = 16
PRECISION_TOTAL = 1 << PRECISION_BITS
N_SIGMA_BINS = 256
SIGMA_MIN = 0.01
SIGMA_MAX = 300.0

# Two CDF window configs:  small for qp1-3, large for qp4-7
SMALL_WIN = 65;  SMALL_CENTER = 32     # covers ±32
LARGE_WIN = 385; LARGE_CENTER = 192    # covers ±192

_caches = {}  # win -> (cdf_list, cdf_lens, offs)
_log_min = np.log(SIGMA_MIN)
_log_max = np.log(SIGMA_MAX)
_inv_step = N_SIGMA_BINS / (_log_max - _log_min)
_pool = ThreadPoolExecutor(max_workers=3)


def _get_cache(win, center):
    if win in _caches:
        return _caches[win]
    edges = np.exp(np.linspace(_log_min, _log_max, N_SIGMA_BINS + 1))
    centers = np.exp(0.5 * (np.log(edges[:-1]) + np.log(edges[1:])))
    k = np.arange(win, dtype=np.float64)
    arg = (k[None, :] - center - 0.5) / (
        (centers[:, None] + 1e-10) * math.sqrt(2))
    cdf_f = 0.5 * (1.0 + scipy_erf(arg))
    cdf = np.zeros((N_SIGMA_BINS, win + 1), dtype=np.int32)
    cdf[:, 1:win] = np.clip(
        np.floor(cdf_f[:, 1:] * PRECISION_TOTAL).astype(np.int32), 0, PRECISION_TOTAL)
    cdf[:, win] = PRECISION_TOTAL
    for i in range(1, win + 1):
        cdf[:, i] = np.maximum(cdf[:, i], cdf[:, i - 1] + 1)
    cdf = np.clip(cdf, 0, PRECISION_TOTAL)
    cdf[:, win] = PRECISION_TOTAL
    for i in range(win - 1, 0, -1):
        cdf[:, i] = np.minimum(cdf[:, i], cdf[:, i + 1] - 1)
    entry = (cdf.tolist(), [win + 1] * N_SIGMA_BINS, [0] * N_SIGMA_BINS, win, center)
    _caches[win] = entry
    return entry


def _qsigma(sigma_flat):
    log_s = np.log(np.clip(sigma_flat, SIGMA_MIN, SIGMA_MAX))
    return np.clip(
        np.floor((log_s - _log_min) * _inv_step).astype(np.int32),
        0, N_SIGMA_BINS - 1)


def _select_win(qp):
    if qp <= 3:
        return SMALL_WIN, SMALL_CENTER
    return LARGE_WIN, LARGE_CENTER


def _enc(symbols, mu_round, sigma_flat, win, center):
    cdf_list, cdf_lens, offs, _, _ = _get_cache(win, center)
    idx = _qsigma(sigma_flat)
    ss = np.clip(symbols.astype(np.int32) - mu_round + center,
                 0, win - 1).astype(np.int32)
    return np_encode_with_indexes(
        np.ascontiguousarray(ss), np.ascontiguousarray(idx),
        cdf_list, cdf_lens, offs)


def _dec(bs, n, mu_round, sigma_flat, win, center):
    cdf_list, cdf_lens, offs, _, _ = _get_cache(win, center)
    idx = _qsigma(sigma_flat)
    ds = np_decode_with_indexes(
        bs, np.ascontiguousarray(idx), cdf_list, cdf_lens, offs)
    return ds + mu_round - center


def _lvdata(ret, lv):
    zr = ret[f"z{lv}_rounded"]
    zs = ret[f"z{lv}_sigma"]
    sh = list(zr.shape)
    sym = (zr + 511).astype(np.int16).reshape(-1).astype(np.int32)
    if lv == 1:
        sf = np.tile(np.reshape(zs, (1, -1, 1, 1)),
                     (sh[0], 1, sh[2], sh[3])).reshape(-1).astype(np.float64)
        mr = np.full(len(sym), 511, dtype=np.int32)
    else:
        sf = zs.reshape(-1).astype(np.float64)
        mr = np.round(ret[f"z{lv}_mu"] + 511).astype(np.int32).reshape(-1)
    return sym, mr, sf, sh


def _dec_params(ret_d, lv, zs):
    if lv == 0:
        s = ret_d["z1_sigma"]
        st = np.tile(np.reshape(s, (1, -1, 1, 1)),
                     (zs[0], 1, zs[2], zs[3])).reshape(-1).astype(np.float64)
        return st, np.full(int(np.prod(zs)), 511, dtype=np.int32)
    k = f"z{lv+1}"
    return (ret_d[f"{k}_sigma"].reshape(-1).astype(np.float64),
            np.round(ret_d[f"{k}_mu"] + 511).astype(np.int32).reshape(-1))


# ---- high-level API ----

def fast_encode_image(net, image_path, qp, model_type, device, parallel=True):
    from networks import get_partition
    win, center = _select_win(qp)
    _get_cache(win, center)

    f = Image.open(image_path).convert("RGB")
    fshape = [f.size[1], f.size[0], 3]
    x = np.array(f).reshape([1] + fshape)
    h, w = (fshape[0] // 64) * 64, (fshape[1] // 64) * 64
    if h < fshape[0]: h += 64
    if w < fshape[1]: w += 64
    pu = (h - fshape[0]) // 2; pd = (h - fshape[0]) - pu
    pl = (w - fshape[1]) // 2; pr = (w - fshape[1]) - pl
    x = np.pad(x, [[0, 0], [pu, pd], [pl, pr], [0, 0]])
    torch_x = (torch.tensor(x).permute(0, 3, 1, 2).float() / 127.5) - 1

    hs, ws = get_partition(h, w)
    t_gpu = t_rans = 0.0
    hdr = bytearray()
    hdr.append((qp << 1) | model_type)
    hdr += np.array([fshape[0], fshape[1]], dtype=np.uint16).tobytes()
    body = bytearray()

    ph = 0
    for th in hs:
        pw = 0
        for tw in ws:
            blk = torch_x[:, :, ph:ph + th, pw:pw + tw].to(device)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.no_grad():
                ret = net.encode(blk)
            torch.cuda.synchronize(); t_gpu += time.perf_counter() - t0

            preps = {lv: _lvdata(ret, lv) for lv in (1, 2, 3)}

            t0 = time.perf_counter()
            if parallel:
                futs = {lv: _pool.submit(_enc, *preps[lv][:3], win, center)
                        for lv in (1, 2, 3)}
                bss = {lv: futs[lv].result() for lv in (1, 2, 3)}
            else:
                bss = {lv: _enc(*preps[lv][:3], win, center) for lv in (1, 2, 3)}
            t_rans += time.perf_counter() - t0

            for lv in (1, 2, 3):
                b = bss[lv]
                body += struct.pack("<I", len(b))
                body += b
            pw += tw
        ph += th
    return bytes(hdr) + bytes(body), t_gpu, t_rans


def fast_decode_image(net, compressed, device):
    from networks import get_partition

    buf = compressed[0]
    qp = buf >> 1
    win, center = _select_win(qp)
    _get_cache(win, center)

    mode = "low" if qp <= 3 else "high"
    n_ch = 192 if mode == "low" else 384
    arr = np.frombuffer(compressed[1:5], dtype=np.uint16)
    oh, ow = int(arr[0]), int(arr[1])
    h, w = (oh // 64) * 64, (ow // 64) * 64
    if h < oh: h += 64
    if w < ow: w += 64
    pu = (h - oh) // 2; pl = (w - ow) // 2
    padded_h = int(math.ceil(h / 16) * 16)
    padded_w = int(math.ceil(w / 16) * 16)
    hs, ws = get_partition(padded_h, padded_w)

    off = 5
    t_gpu = t_rans = 0.0
    tar = np.zeros((1, 3, padded_h, padded_w), dtype=np.float32)
    ph = 0
    for th in hs:
        pw = 0
        for tw in ws:
            bph = int(math.ceil(th / 16) * 16)
            bpw = int(math.ceil(tw / 16) * 16)
            z_shapes = [
                (1, 128, bph // 64, bpw // 64),
                (1, 256, bph // 32, bpw // 32),
                (1, n_ch, bph // 16, bpw // 16),
            ]
            z_hats = [None] * 3

            for lv in range(3):
                bsl = struct.unpack_from("<I", compressed, off)[0]; off += 4
                bitstream = compressed[off:off + bsl]; off += bsl

                torch.cuda.synchronize(); t0 = time.perf_counter()
                with torch.no_grad():
                    if lv == 0:
                        ret_d = net.decode(None, 0)
                torch.cuda.synchronize(); t_gpu += time.perf_counter() - t0

                sf, mr = _dec_params(ret_d, lv, z_shapes[lv])

                t0 = time.perf_counter()
                decoded = _dec(bitstream, int(np.prod(z_shapes[lv])), mr, sf,
                               win, center)
                t_rans += time.perf_counter() - t0

                z_hats[lv] = (decoded.astype(np.float32) - 511).reshape(z_shapes[lv])

                torch.cuda.synchronize(); t0 = time.perf_counter()
                with torch.no_grad():
                    if lv == 0:
                        ret_d = net.decode(
                            {"z1_rounded": torch.tensor(z_hats[0]).to(device)}, 1)
                    elif lv == 1:
                        ret_d = net.decode(
                            {"z1_rounded": torch.tensor(z_hats[0]).to(device),
                             "z2_rounded": torch.tensor(z_hats[1]).to(device)}, 2)
                    else:
                        x_tilde = net.decode(
                            {"z1_rounded": torch.tensor(z_hats[0]).to(device),
                             "z2_rounded": torch.tensor(z_hats[1]).to(device),
                             "z3_rounded": torch.tensor(z_hats[2]).to(device)}, 3)
                torch.cuda.synchronize(); t_gpu += time.perf_counter() - t0

            tar[:, :, ph:ph + th, pw:pw + tw] = x_tilde
            pw += tw
        ph += th

    r = tar[0].astype(np.uint8)
    r = r.transpose((1, 2, 0))[pu:pu + oh, pl:pl + ow, :]
    return Image.fromarray(r), t_gpu, t_rans


def fast_encode_decode(image_path, output_path, qp, model_type, device_str,
                       net=None, verbose=True):
    import networks as N_mod
    device = torch.device(device_str)
    N_mod.device = device

    if net is None:
        NetCls = N_mod.NetLow if qp <= 3 else N_mod.NetHigh
        net = NetCls().eval().to(device)
        td = N_mod.load_tf_weights(net.state_dict(),
                                   f"models/model{model_type}_qp{qp}.pk")
        net.load_state_dict(td)
        with torch.no_grad():
            _ = net.encode(torch.randn(1, 3, 64, 64).to(device))
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    comp, eg, er = fast_encode_image(net, image_path, qp, model_type, device)
    recon, dg, dr = fast_decode_image(net, comp, device)
    total = time.perf_counter() - t0

    if output_path:
        recon.save(output_path)
    if verbose:
        print(f"  Encode:  GPU {eg*1e3:6.1f}ms  rANS {er*1e3:6.1f}ms")
        print(f"  Decode:  GPU {dg*1e3:6.1f}ms  rANS {dr*1e3:6.1f}ms")
        print(f"  TOTAL:   {total*1e3:6.1f}ms  "
              f"(GPU {(eg+dg)*1e3:.1f} + rANS {(er+dr)*1e3:.1f})  "
              f"Size: {len(comp)} B")
    return recon, comp, total
