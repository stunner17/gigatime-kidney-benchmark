"""Which pixel size does GigaTIME expect? Test 0.25 vs 0.5 um/px on one patient, using virtual DAPI only.

The paper describes 128 um patches as 256 px (0.5 um/px); the authors' code resizes 556 px per 128 um to 512 px
(about 0.25 um/px). Neither choice uses a protein outcome here: virtual DAPI is compared with
  dapi_vs_hema   haematoxylin from colour deconvolution of the same crop (Pearson, tissue pixels, 2 um smoothing)
  dapi_vs_cells  MIST cell count per 16 um bin (Pearson), with a control: the map shifted by 256 um
on three 512 um crops (immune-rich, aSMA-rich, most cells) of a patient whose slide is the MIST section.

usage:  python scripts/01_pixel_size.py --data data --patient P06 --stain PAS --out results/pixel_size.csv
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models as M  # noqa
import slides as S  # noqa

CROP_UM = 512.0


def pear(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(pearsonr(a[ok], b[ok])[0]) if ok.sum() >= 10 and np.std(a[ok]) and np.std(b[ok]) else np.nan


def pick_crops(Y, X, V):
    """three non-overlapping 512 um windows: immune-rich (CD20+CD4+CD8a), aSMA-rich, most cells"""
    w = CROP_UM / S.MIST_UM_PER_PX
    cand = []
    for r in np.arange(0, Y.max() - w + 1, w / 2):
        for c in np.arange(0, X.max() - w + 1, w / 2):
            m = (Y >= r) & (Y < r + w) & (X >= c) & (X < c + w)
            if m.sum() >= 300:
                cand.append(dict(row0=r, col0=c, n=int(m.sum()), immune=V.loc[m, ['CD20', 'CD4', 'CD8a']].mean().mean(),
                                 asma=V.loc[m, 'aSMA'].mean()))
    d = pd.DataFrame(cand)
    for k in ('immune', 'asma'):
        d[k] = (d[k] - d[k].mean()) / (d[k].std() + 1e-9)
    chosen = []
    for name, key in (('immune', 'immune'), ('asma', 'asma'), ('dense', 'n')):
        for _, r in d.sort_values(key, ascending=False).iterrows():
            if all(abs(r.row0 - q['row0']) >= w or abs(r.col0 - q['col0']) >= w for q in chosen):
                chosen.append(dict(name=name, row0=r.row0, col0=r.col0)); break
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    ap.add_argument('--patient', required=True)
    ap.add_argument('--stain', default='PAS')
    ap.add_argument('--out', default='results/pixel_size.csv')
    a = ap.parse_args()
    from skimage.color import rgb2hed
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nets = {'gigatime': (M.load_gigatime(M.REPOS['gigatime'], dev)[0], 512),
            'gigatime_flash': (M.load_gigatime_flash(M.REPOS['gigatime_flash'], dev)[0], 256)}
    c = S.read_cells(a.data, a.patient)
    Y, X = c['Global Y'].values.astype(float), c['Global X'].values.astype(float)
    V = np.log1p(c[S.PROTEINS].clip(lower=0))
    A, smpp = S.read_transform(a.data, a.patient, a.stain)
    img, H, W = S.open_rgb(os.path.join(a.data, a.patient, f'{a.stain}.tif'))
    rows = []
    for cr in pick_crops(Y, X, V):
        for s in (0.25, 0.5):
            n = int(round(CROP_UM / s)); step = s / S.MIST_UM_PER_PX
            rgb = S.sample_on_mist_grid(img, H, W, A, cr['row0'], cr['col0'], n, s, smpp)
            tis = rgb.astype(np.float32).mean(-1) < 220
            m = (Y >= cr['row0']) & (Y < cr['row0'] + n * step) & (X >= cr['col0']) & (X < cr['col0'] + n * step)
            ci, cj = (Y[m] - cr['row0']) / step - 0.5, (X[m] - cr['col0']) / step - 0.5
            g = max(1, int(round(4 / s)))
            hema = ndi.gaussian_filter(rgb2hed(rgb)[..., 0].astype(np.float32), 2 / s)[::g, ::g]
            for name, (net, window) in nets.items():
                dapi = M.virtual_mif(net, rgb, window, dev)[M.CHANNELS.index('DAPI')]
                t = tis[::g, ::g]
                rows.append(dict(crop=cr['name'], um_per_px=s, model=name, check='dapi_vs_hema', control=False,
                                 r=pear(ndi.gaussian_filter(dapi, 2 / s)[::g, ::g][t], hema[t])))
                bs = int(round(16 / s)); nb = n // bs
                cnt = np.bincount(np.clip((ci // bs).astype(int), 0, nb - 1) * nb +
                                  np.clip((cj // bs).astype(int), 0, nb - 1), minlength=nb * nb)
                tb = tis[:nb * bs, :nb * bs].reshape(nb, bs, nb, bs).mean((1, 3)).ravel() > 0.5
                for ctrl in (False, True):
                    dm = np.roll(dapi, (n // 2, n // 2), (0, 1)) if ctrl else dapi
                    dbin = dm[:nb * bs, :nb * bs].reshape(nb, bs, nb, bs).mean((1, 3)).ravel()
                    rows.append(dict(crop=cr['name'], um_per_px=s, model=name, check='dapi_vs_cells', control=ctrl,
                                     r=pear(dbin[tb], cnt[tb].astype(float))))
    d = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    d.to_csv(a.out, index=False)
    print(d.groupby(['check', 'control', 'model', 'um_per_px']).r.median().unstack().round(3))


if __name__ == '__main__':
    main()
