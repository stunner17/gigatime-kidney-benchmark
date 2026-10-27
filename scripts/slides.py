"""Reading slides and MIST cells, and sampling a stained slide onto the MIST grid.

Expected data layout (one folder per patient):
    data/<patient>/cells.csv     MIST cells: 'Global X', 'Global Y' (MIST pixels) and one column per protein
    data/<patient>/<STAIN>.tif   RGB whole-slide image (OME-TIFF), e.g. PAS.tif, HE.tif
    data/<patient>/<STAIN>.json  registration: {"A_mist_to_stain": 2x3 affine, "stain_um_per_px": float}
                                 A maps MIST (row = Global Y, col = Global X) to stain (row, col)
    data/same_section.txt        patients whose PAS slide is the MIST section itself, one per line
"""
import os, json
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

MIST_UM_PER_PX = 0.32495
PROTEINS = ['ALCAM', 'WT1', 'STAT1', 'MCP1', 'Megalin', 'IL 23', 'CD56', 'IgG4', 'VCAM-1', 'CD8a', 'CD163', 'CD4',
            'C3D', 'VDBP', 'CD20', 'Granzyme B', 'C1q', 'AQP2', 'aSMA', 'Collagen IV', 'Vimentin']


def patients(data, stain):
    return sorted(p for p in os.listdir(data) if os.path.exists(os.path.join(data, p, f'{stain}.json')))


def same_section(data):
    f = os.path.join(data, 'same_section.txt')
    return {l.strip() for l in open(f) if l.strip()} if os.path.exists(f) else set()


def read_cells(data, pid):
    return pd.read_csv(os.path.join(data, pid, 'cells.csv'))


def read_transform(data, pid, stain):
    rec = json.load(open(os.path.join(data, pid, f'{stain}.json')))
    return np.array(rec['A_mist_to_stain'], np.float64), float(rec['stain_um_per_px'])


def open_rgb(path):
    """full-resolution RGB plane as (H, W, 3) uint8: memory-mapped when uncompressed, decoded otherwise"""
    import tifffile
    t = tifffile.TiffFile(path); p = t.pages[0]; H, W = p.imagelength, p.imagewidth
    o, b = p.dataoffsets, p.databytecounts
    if p.compression == 1 and all(o[i] + b[i] == o[i + 1] for i in range(len(o) - 1)) \
            and p.samplesperpixel == 3 and p.dtype == np.uint8:
        t.close()
        return np.memmap(path, dtype=np.uint8, mode='r', offset=o[0], shape=(H, W, 3)), H, W
    arr = p.asarray(); t.close()
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.dtype != np.uint8:
        arr = (arr / max(1, arr.max()) * 255).astype(np.uint8)
    return arr[..., :3], H, W


def sample_on_mist_grid(img, H, W, A, row0, col0, n, um_per_px, stain_um_per_px):
    """n x n RGB image in MIST orientation: output pixel (i, j) is MIST pixel (row0 + (i + .5) * step, ...), with
    step = um_per_px / MIST_UM_PER_PX, read from the slide through A (bilinear). White outside the slide."""
    g = (np.arange(n) + 0.5) * um_per_px / MIST_UM_PER_PX
    r, c = np.meshgrid(row0 + g, col0 + g, indexing='ij')
    pr = A[0, 0] * r + A[0, 1] * c + A[0, 2]; pc = A[1, 0] * r + A[1, 1] * c + A[1, 2]
    out = np.full((n, n, 3), 255, np.uint8)
    inside = (pr >= 0) & (pr <= H - 1) & (pc >= 0) & (pc <= W - 1)
    if not inside.any():
        return out
    r0, r1 = int(max(0, np.floor(pr[inside].min()) - 2)), int(min(H, np.ceil(pr[inside].max()) + 3))
    c0, c1 = int(max(0, np.floor(pc[inside].min()) - 2)), int(min(W, np.ceil(pc[inside].max()) + 3))
    crop = np.asarray(img[r0:r1, c0:c1, :], dtype=np.float32)
    ratio = um_per_px / stain_um_per_px
    for k in range(3):
        ch = crop[..., k]
        if ratio > 1.3:                                  # downsampling: smooth first to avoid aliasing
            ch = ndi.gaussian_filter(ch, 0.4 * ratio)
        v = ndi.map_coordinates(ch, [pr - r0, pc - c0], order=1, mode='constant', cval=255)
        out[..., k] = np.clip(np.round(v), 0, 255).astype(np.uint8)
    out[~inside] = 255
    return out


def tissue_mask(rgb):
    from skimage.morphology import disk
    m = rgb.astype(np.float32).mean(-1) < 220
    return ndi.binary_fill_holes(ndi.binary_closing(m, disk(3)))
