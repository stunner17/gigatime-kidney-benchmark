"""Run GigaTIME, GigaTIME-Flash and Prov-GigaPath on every 320 um MIST square of every slide.

Squares: the MIST grid cut into 320 um squares (985 MIST px); a square is kept if it holds >= 30 MIST cells.
Each square is sampled from the slide onto the MIST grid at 0.25 um/px with a 64 um margin (1792 x 1792 px).
  GigaTIME        512 px windows, stride 256 | overlapping windows blended with Hann-squared weights
  GigaTIME-Flash  256 px windows, stride 128 |
                  -> per square (tissue pixels of the 320 um core): mean probability and fraction > 0.5, 23 channels
                  -> per MIST cell: map value at the cell centre (3 um smoothing)
  Prov-GigaPath:  3 x 3 tiles of 224 px at 0.5 um/px covering the core -> mean embedding (1536)
Also saved: per-square ground truth (mean log1p of each MIST protein) and the square's tissue fraction.

Output: <out>/squares/<patient>_<STAIN>.npz and <out>/cells/<patient>_<STAIN>.npz. Finished slides are skipped.
usage:  python scripts/02_run_models.py --data data --out results/models [--stains PAS,HE]   (GPU recommended)
"""
import os, sys, time, argparse
import numpy as np
import torch
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models as M  # noqa
import slides as S  # noqa

REG = int(round(320 / S.MIST_UM_PER_PX))         # 985 MIST px = 320 um
UM = 0.25                                        # GigaTIME input pixel size (see 01_pixel_size.py)
CORE, PAD = 1280, 256                            # 320 um core, 64 um margin at 0.25 um/px
N = CORE + 2 * PAD
MINC = 30


def hann2(w):
    v = np.sin(np.pi * (np.arange(w) + 0.5) / w) ** 2
    return np.outer(v, v).astype(np.float32)


@torch.no_grad()
def blended(net, rgb, window, stride, dev, batch):
    """(23, H, W) sigmoid maps from overlapping windows, weighted by Hann-squared and normalised"""
    H, W, _ = rgb.shape
    pos = [(i, j) for i in range(0, H - window + 1, stride) for j in range(0, W - window + 1, stride)]
    wt = torch.from_numpy(hann2(window)).to(dev)
    acc = torch.zeros(23, H, W, device=dev); wsum = torch.zeros(H, W, device=dev)
    x_all = M.to_tensor(rgb[None])[0]
    for s in range(0, len(pos), batch):
        pp = pos[s:s + batch]
        x = torch.stack([x_all[:, i:i + window, j:j + window] for i, j in pp]).to(dev)
        y = torch.sigmoid(net(x).float())
        for k, (i, j) in enumerate(pp):
            acc[:, i:i + window, j:j + window] += y[k] * wt; wsum[i:i + window, j:j + window] += wt
    return (acc / wsum.clamp_min(1e-6)).cpu().numpy()


@torch.no_grad()
def embed(net, tiles, mean, std, dev):
    x = (tiles.astype(np.float32) / 255.0 - mean) / std
    x = torch.from_numpy(x.transpose(0, 3, 1, 2).copy()).to(dev)
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=dev.type == 'cuda'):
        e = net(x)
    return e.float().mean(0).cpu().numpy()


def run_slide(data, pid, stain, nets, dev, max_squares=0):
    A, stain_mpp = S.read_transform(data, pid, stain)
    c = S.read_cells(data, pid)
    prot = [p for p in S.PROTEINS if p in c.columns]
    Y, X = c['Global Y'].values.astype(np.float64), c['Global X'].values.astype(np.float64)
    V = np.log1p(np.clip(c[prot].values.astype(np.float32), 0, None))
    key = (Y // REG).astype(int) * 100000 + (X // REG).astype(int)
    uk, cnt = np.unique(key, return_counts=True)
    keep = uk[cnt >= MINC]
    if max_squares:
        keep = keep[:max_squares]
    img, H, W = S.open_rgb(os.path.join(data, pid, f'{stain}.tif'))
    step = UM / S.MIST_UM_PER_PX
    Q = dict(sq_row=[], sq_col=[], n_cells=[], tissue=[], gt=[], gigatime_mean=[], gigatime_pos=[],
             gigatime_flash_mean=[], gigatime_flash_pos=[], gigapath=[])
    C = dict(cell=[], sq=[], gt=[], gigatime=[], gigatime_flash=[])
    for q, kk in enumerate(keep):
        r_, c_ = divmod(int(kk), 100000)
        rgb = S.sample_on_mist_grid(img, H, W, A, r_ * REG - PAD * step, c_ * REG - PAD * step, N, UM, stain_mpp)
        core = rgb[PAD:PAD + CORE, PAD:PAD + CORE]
        tis = S.tissue_mask(core)
        idx = np.nonzero(key == kk)[0]
        Q['sq_row'].append(r_); Q['sq_col'].append(c_); Q['tissue'].append(float(tis.mean()))
        Q['n_cells'].append(len(idx)); Q['gt'].append(V[idx].mean(0))
        ci = (Y[idx] - r_ * REG) / step - 0.5; cj = (X[idx] - c_ * REG) / step - 0.5
        C['cell'].append(idx); C['sq'].append(np.full(len(idx), q)); C['gt'].append(V[idx].astype(np.float16))
        tm = tis if tis.any() else np.ones_like(tis)
        for k, (window, stride, b) in (('gigatime', (512, 256, 12)), ('gigatime_flash', (256, 128, 64))):
            p = blended(nets[k], rgb, window, stride, dev, b)[:, PAD:PAD + CORE, PAD:PAD + CORE]
            Q[f'{k}_mean'].append(p[:, tm].mean(1)); Q[f'{k}_pos'].append((p[:, tm] > 0.5).mean(1))
            sm = ndi.gaussian_filter(p.reshape(23, CORE // 4, 4, CORE // 4, 4).mean((2, 4)), (0, 3, 3))  # 1 um/px
            di, dj = (ci + 0.5) / 4 - 0.5, (cj + 0.5) / 4 - 0.5
            C[k].append(np.stack([ndi.map_coordinates(sm[ch], [di, dj], order=1, mode='nearest')
                                  for ch in range(23)], 1).astype(np.float16))
        half = core.reshape(640, 2, 640, 2, 3).mean((1, 3)).astype(np.uint8)        # 0.5 um/px
        tiles = np.stack([half[i:i + 224, j:j + 224] for i in (0, 208, 416) for j in (0, 208, 416)])
        Q['gigapath'].append(embed(nets['gigapath'], tiles, M.MEAN, M.STD, dev).astype(np.float16))
    return Q, C, prot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    ap.add_argument('--out', default='results/models')
    ap.add_argument('--stains', default='PAS,HE')
    ap.add_argument('--max_squares', type=int, default=0, help='testing only')
    a = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nets = {'gigatime': M.load_gigatime(M.REPOS['gigatime'], dev)[0],
            'gigatime_flash': M.load_gigatime_flash(M.REPOS['gigatime_flash'], dev)[0],
            'gigapath': M.load_gigapath(M.REPOS['gigapath'], dev)[0]}
    for sub in ('squares', 'cells'):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    for stain in a.stains.split(','):
        for pid in S.patients(a.data, stain):
            f = os.path.join(a.out, 'squares', f'{pid}_{stain}.npz')
            if os.path.exists(f):
                continue
            t0 = time.time()
            Q, C, prot = run_slide(a.data, pid, stain, nets, dev, a.max_squares)
            meta = dict(proteins=np.array(prot), channels=np.array(M.CHANNELS))
            np.savez_compressed(f, **meta, **{k: np.asarray(v) for k, v in Q.items()})
            np.savez_compressed(os.path.join(a.out, 'cells', f'{pid}_{stain}.npz'), **meta,
                                **{k: np.concatenate(v) for k, v in C.items()})
            print(f'{pid} {stain}: {len(Q["sq_row"])} squares, {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
