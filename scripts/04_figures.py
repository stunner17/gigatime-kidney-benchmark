"""Figures 1-4 from the result tables written by 03_evaluate.py (Arial, 300 dpi, PNG + SVG).

usage:  python scripts/04_figures.py --results results --out figures
"""
import os, argparse
import numpy as np
import pandas as pd

RES, OUT = 'results', 'figures'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'Arial', 'font.size': 7, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'svg.fonttype': 'none'})
NAMES = {'gigatime': 'GigaTIME', 'flash': 'GigaTIME-Flash', 'both': 'GigaTIME + Flash',
         'gigapath': 'Prov-GigaPath'}
COL = {'gigatime': '#b8b8b8', 'flash': '#b8b8b8', 'both': '#b8b8b8', 'gigapath': '#2b6cb0'}
STAIN = {'PAS': 'PAS (same section as MIST)', 'HE': 'H&E (different section)'}
RNG = np.random.default_rng(0)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for e in ('png', 'svg'):
        fig.savefig(os.path.join(OUT, f'{name}.{e}'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def f1():
    summ = pd.read_csv(os.path.join(RES, 'B_trained_summary.csv'))
    feats = ['gigatime', 'flash', 'both', 'gigapath']
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for ax, s in zip(axes, ('PAS', 'HE')):
        pp = pd.read_csv(os.path.join(RES, f'B_trained_per_patient_{s}.csv'), index_col=0)
        for i, f in enumerate(feats):
            v = pp[f].values
            ax.scatter(i + RNG.uniform(-.13, .13, len(v)), v, s=7, color=COL[f], lw=0, alpha=.9)
            ax.hlines(np.median(v), i - .32, i + .32, color='black', lw=1.4)
            n95 = summ[(summ.stain == s) & (summ.features == f)].null_95.iloc[0]
            ax.hlines(n95, i - .32, i + .32, color='#c53030', lw=.9, ls=(0, (2, 1.5)))
        ax.axhline(0, color='0.85', lw=.5, zorder=0)
        ax.set_xticks(range(len(feats))); ax.set_xticklabels([NAMES[f] for f in feats], rotation=30, ha='right')
        ax.set_title(f'{STAIN[s]}, n = {len(pp)} patients', fontsize=7.5)
    axes[0].set_ylabel('Held-out patient score\n(median Spearman over 21 proteins)')
    axes[1].plot([], [], color='black', lw=1.4, label='median over patients')
    axes[1].plot([], [], color='#c53030', lw=.9, ls=(0, (2, 1.5)), label='shuffle null, 95th percentile')
    fig.tight_layout()
    fig.legend(*axes[1].get_legend_handles_labels(), frameon=False, fontsize=6, ncol=2, loc='upper center',
               bbox_to_anchor=(0.5, 0.0))
    save(fig, 'F1_trained')


def f2():
    z = pd.read_csv(os.path.join(RES, 'A_direct_per_patient.csv'))
    zs = pd.read_csv(os.path.join(RES, 'A_direct_summary.csv'))
    z = z[(z.kind == 'direct') & (z.summary == 'pos')]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
    for ax, s in zip(axes, ('PAS', 'HE')):
        x0 = 0; ticks, labs = [], []
        for m in ('gigatime', 'flash'):
            for ch in ('CD4', 'CD8', 'CD20'):
                g = z[(z.stain == s) & (z.model == m) & (z.channel == ch)]
                for k, (col, c) in enumerate((('rho', '#2b6cb0'), ('rho_shift640', '#bdbdbd'))):
                    v = g[col].values
                    ax.scatter(x0 + k * .38 + RNG.uniform(-.08, .08, len(v)), v, s=5, color=c, lw=0)
                    ax.hlines(np.nanmedian(v), x0 + k * .38 - .15, x0 + k * .38 + .15, color='black', lw=1.2)
                r = zs[(zs.stain == s) & (zs.model == m) & (zs.channel == ch) & (zs.summary == 'pos')].iloc[0]
                if r.p_holm < 0.05:
                    ax.text(x0 + .19, 0.78, f'Holm p = {r.p_holm:.3f}', ha='center', fontsize=5.5)
                ticks.append(x0 + .19); labs.append(f"{'GigaTIME' if m == 'gigatime' else 'Flash'}\n{ch}"); x0 += 1.2
        ax.axhline(0, color='0.85', lw=.5, zorder=0)
        ax.set_xticks(ticks); ax.set_xticklabels(labs, fontsize=5.8)
        ax.set_title(STAIN[s], fontsize=7.5); ax.set_ylim(-.9, .9)
    axes[0].set_ylabel('Spearman, virtual vs MIST\n(per patient, 320 um squares)')
    axes[1].scatter([], [], color='#2b6cb0', s=8, label='true position'); axes[1].scatter([], [], color='#bdbdbd', s=8,
                                                                                          label='640 um shift control')
    axes[1].legend(frameon=False, fontsize=6, loc='lower right')
    fig.tight_layout(); save(fig, 'F2_direct')


def f3():
    p = pd.read_csv(os.path.join(RES, 'B_per_protein.csv'))
    p = p[p.stain == 'PAS'].set_index('protein').drop(columns='stain').sort_values('gigapath')
    fig, ax = plt.subplots(figsize=(3.3, 3.9))
    y = np.arange(len(p))
    ax.hlines(y, p[['gigapath', 'both']].min(axis=1), p[['gigapath', 'both']].max(axis=1), color='0.85', lw=.8)
    for f, mk in (('both', 'D'), ('gigapath', 's')):
        ax.scatter(p[f], y, marker=mk, s=13, color=COL[f] if f != 'both' else '#9e9e9e', label=NAMES[f], zorder=3,
                   lw=0)
    ax.axvline(0, color='0.85', lw=.5, zorder=0)
    ax.set_yticks(y); ax.set_yticklabels(p.index)
    ax.set_xlabel('Spearman, median over 17 held-out patients')
    ax.set_title('Per protein, PAS', fontsize=7.5)
    ax.legend(frameon=False, fontsize=6, loc='lower right')
    fig.tight_layout(); save(fig, 'F3_per_protein')


def f4():
    mut = pd.read_csv(os.path.join(RES, 'collapse_mutual.csv'))
    tr = pd.read_csv(os.path.join(RES, 'collapse_tracking_PAS.csv'), index_col=0)
    prots = list(tr.columns)
    fig = plt.figure(figsize=(7.0, 3.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.6], wspace=.35)
    ax = fig.add_subplot(gs[0])
    w = mut[mut.stain == 'PAS']
    for i, (src, c) in enumerate((('mist', '#4a4a4a'), ('gigatime', '#2b6cb0'), ('flash', '#63b3ed'))):
        v = w[w.source == src].mutual_rho.values
        ax.scatter(i + RNG.uniform(-.12, .12, len(v)), v, s=7, color=c, lw=0)
        ax.hlines(np.median(v), i - .3, i + .3, color='black', lw=1.4)
    ax.axhline(0, color='0.85', lw=.5, zorder=0)
    ax.set_xticks(range(3)); ax.set_xticklabels(['MIST\n(measured)', 'GigaTIME\n(virtual)', 'Flash\n(virtual)'])
    ax.set_ylabel('Mean Spearman among CD4, CD8, CD20\nacross squares (per patient)')
    ax.set_title('a  Immune channels move together', fontsize=7.5, loc='left')
    ax.set_ylim(-.4, 1.05)
    ax = fig.add_subplot(gs[1])
    keys = [('PAS', m, ch) for m in ('gigatime', 'flash') for ch in ('DAPI', 'CD3', 'CD4', 'CD8', 'CD20', 'CD68')]
    M = np.array([tr.loc[f'{m}:{ch}'].values for _, m, ch in keys])
    order = np.argsort(-np.abs(M).mean(0))[:12]
    im = ax.imshow(M[:, order], cmap='RdBu_r', vmin=-.6, vmax=.6, aspect='auto')
    ax.set_xticks(range(len(order))); ax.set_xticklabels([prots[j] for j in order], rotation=45, ha='right')
    ax.set_yticks(range(len(keys))); ax.set_yticklabels([f"{'GigaTIME' if m == 'gigatime' else 'Flash'} {ch}"
                                                         for _, m, ch in keys])
    for (i, j), v in np.ndenumerate(M[:, order]):
        if abs(v) >= .25:
            ax.text(j, i, f'{v:.2f}'.replace('0.', '.'), ha='center', va='center', fontsize=5,
                    color='white' if abs(v) > .45 else 'black')
    ax.axhline(5.5, color='white', lw=1.5)
    cb = fig.colorbar(im, ax=ax, fraction=.025, pad=.02); cb.set_label('median Spearman (17 patients)', fontsize=6)
    cb.outline.set_linewidth(.4)
    ax.set_title('b  What each virtual channel tracks in MIST (PAS)', fontsize=7.5, loc='left')
    save(fig, 'F4_collapse')


def main():
    global RES, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results')
    ap.add_argument('--out', default='figures')
    a = ap.parse_args()
    RES, OUT = a.results, a.out
    f1(); f2(); f3(); f4()
    print('figures in', OUT)


if __name__ == '__main__':
    main()
