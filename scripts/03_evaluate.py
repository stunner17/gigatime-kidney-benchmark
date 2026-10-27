"""Evaluate the model outputs against MIST: direct comparison, trained comparison, and the compartment collapse.

Squares used: >= 30 MIST cells and >= 60% stain tissue. Ground truth = mean log1p protein per 320 um square,
z-scored per protein within each patient. All scores are Spearman across squares within a patient.

A. Direct comparison (no training on MIST): each GigaTIME / Flash virtual channel vs the matching MIST protein (direct: CD4, CD8-CD8a, CD20;
   proxy: Actin-D / Transgelin - aSMA, CD68 - CD163), per-square mean probability and fraction > 0.5. Control: the
   same value paired with the ground truth of a square 640 um away. Paired Wilcoxon over patients, Holm over all tests.
   Same-section PAS also at cell level.
B. Trained comparison: ridge per protein on frozen features (GigaTIME 46, Flash 46, both 92, Prov-GigaPath 1536),
   leave one patient out, alpha by RidgeCV on training patients. Score per held-out patient = median over the
   21 proteins. Null: ground truth shuffled within the held-out patient, 200 times. Model comparisons: paired
   Wilcoxon over patients, Holm.
C. Collapse: mutual Spearman of virtual vs measured CD4, CD8, CD20 within each patient, and what each virtual channel
   tracks among the 21 MIST proteins.

usage:  python scripts/03_evaluate.py --models results/models --data data --out results
"""
import os, glob, argparse, itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

IN, SAME, DATA = None, set(), {}
PAIRS = [('CD4', 'CD4', 'direct'), ('CD8', 'CD8a', 'direct'), ('CD20', 'CD20', 'direct'),
         ('Actin-D', 'aSMA', 'proxy'), ('Transgelin', 'aSMA', 'proxy'), ('CD68', 'CD163', 'proxy')]
MINT, MINC = 0.6, 30
ALPHAS = np.logspace(-1, 5, 13)
NPERM = 200
RNG = np.random.default_rng(0)


def sp(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return np.nan
    return spearmanr(a[ok], b[ok])[0]


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m)
    run = 0
    for k, i in enumerate(o):
        run = max(run, min(1, (m - k) * p[i])); adj[i] = run
    return adj



# ---------------------------------------------------------------- data
def load(stain):
    """dict pid -> table of usable squares (features + z-scored ground truth)"""
    D = {}
    for f in sorted(glob.glob(os.path.join(IN, 'squares', f'*_{stain}.npz'))):
        pid = os.path.basename(f)[:-len(f'_{stain}.npz')]
        d = np.load(f, allow_pickle=False)
        ok = (d['tissue'] >= MINT) & (d['n_cells'] >= MINC)
        if ok.sum() < 8:
            continue
        gt = d['gt'][ok].astype(np.float64)
        gt = (gt - gt.mean(0)) / (gt.std(0) + 1e-9)
        D[pid] = dict(row=d['sq_row'][ok], col=d['sq_col'][ok], gt=gt, proteins=list(d['proteins']),
                      channels=list(d['channels']),
                      gigatime=np.c_[d['gigatime_mean'][ok], d['gigatime_pos'][ok]].astype(np.float64),
                      flash=np.c_[d['gigatime_flash_mean'][ok], d['gigatime_flash_pos'][ok]].astype(np.float64),
                      gigapath=d['gigapath'][ok].astype(np.float64))
        D[pid]['both'] = np.c_[D[pid]['gigatime'], D[pid]['flash']]
    return D


def shifted_index(row, col, k=2):
    """for each square, the index of a kept square k squares away (down, up, right, left), else -1"""
    pos = {(r, c): i for i, (r, c) in enumerate(zip(row, col))}
    out = np.full(len(row), -1)
    for i, (r, c) in enumerate(zip(row, col)):
        for dr, dc in ((k, 0), (-k, 0), (0, k), (0, -k)):
            j = pos.get((r + dr, c + dc))
            if j is not None:
                out[i] = j; break
    return out


# ---------------------------------------------------------------- A. direct comparison
def direct(D, stain):
    rows = []
    for pid, d in D.items():
        sh = shifted_index(d['row'], d['col'])
        for model in ('gigatime', 'flash'):
            nch = len(d['channels'])
            for ch, prot, kind in PAIRS:
                if prot not in d['proteins']:
                    continue
                k = d['channels'].index(ch); y = d['gt'][:, d['proteins'].index(prot)]
                for summ, col in (('mean', k), ('pos', nch + k)):
                    x = d[model][:, col]
                    m = sh >= 0
                    rows.append(dict(stain=stain, patient=pid, model=model, channel=ch, protein=prot, kind=kind,
                                     summary=summ, rho=sp(x, y), rho_shift640=sp(x[m], y[sh[m]]), n=len(y)))
    return pd.DataFrame(rows)


def direct_cells(stain):
    """same-section PAS: Spearman across cells within patient; control = cell values of the square 2 away"""
    rows = []
    for f in sorted(glob.glob(os.path.join(IN, 'cells', f'*_{stain}.npz'))):
        pid = os.path.basename(f)[:-len(f'_{stain}.npz')]
        if pid not in SAME:
            continue
        c = np.load(f); s = np.load(os.path.join(IN, 'squares', f'{pid}_{stain}.npz'))
        okq = (s['tissue'] >= MINT) & (s['n_cells'] >= MINC)
        keep = okq[c['sq']]
        prot, chs = list(c['proteins']), list(c['channels'])
        for model in ('gigatime', 'gigatime_flash'):
            for ch, p, kind in PAIRS:
                if p not in prot:
                    continue
                x = c[model][keep, chs.index(ch)].astype(np.float64)
                y = c['gt'][keep, prot.index(p)].astype(np.float64)
                rows.append(dict(patient=pid, model='flash' if model == 'gigatime_flash' else model, channel=ch,
                                 protein=p, kind=kind, rho_cells=sp(x, y), n_cells=int(keep.sum())))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- B. trained ridge
def lopo(D, feat):
    """leave one patient out; returns per-patient median-over-proteins score, its shuffle null, per-protein rho"""
    pids = list(D)
    P = len(D[pids[0]]['proteins'])
    per_pat, per_prot, nulls = {}, [], {}
    for p in pids:
        tr = [q for q in pids if q != p]
        Xtr = np.vstack([D[q][feat] for q in tr]); Ytr = np.vstack([D[q]['gt'] for q in tr])
        sc = StandardScaler().fit(Xtr)
        mdl = RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit(sc.transform(Xtr), Ytr)
        pred = mdl.predict(sc.transform(D[p][feat])); y = D[p]['gt']
        r = np.array([sp(pred[:, j], y[:, j]) for j in range(P)])
        per_pat[p] = np.nanmedian(r)
        per_prot += [dict(patient=p, protein=D[p]['proteins'][j], rho=r[j]) for j in range(P)]
        nl = []
        for _ in range(NPERM):
            perm = RNG.permutation(len(y))
            nl.append(np.nanmedian([sp(pred[:, j], y[perm, j]) for j in range(P)]))
        nulls[p] = np.array(nl)
    return per_pat, pd.DataFrame(per_prot), nulls


def trained(D, stain, feats=('gigatime', 'flash', 'both', 'gigapath')):
    res, prot_rows, null_rows = {}, [], []
    for f in feats:
        pp, pr, nl = lopo(D, f)
        res[f] = pp
        pr['features'] = f; pr['stain'] = stain; prot_rows.append(pr)
        med = np.median(list(pp.values()))
        null_med = np.array([np.median([nl[p][i] for p in pp]) for i in range(NPERM)])
        null_rows.append(dict(stain=stain, features=f, n_patients=len(pp), median=med,
                              mean=np.mean(list(pp.values())), null_mean=null_med.mean(),
                              null_95=np.percentile(null_med, 95),
                              p_vs_null=(1 + (null_med >= med).sum()) / (1 + NPERM),
                              patients_above_null95=int(sum(pp[p] > np.percentile(nl[p], 95) for p in pp))))
        print(f'  {stain:8s} {f:9s} median {med:.3f}  null {null_med.mean():.3f}', flush=True)
    per_patient = pd.DataFrame(res); per_patient.index.name = 'patient'
    comp = []
    for a, b in itertools.combinations(feats, 2):
        d = per_patient[a] - per_patient[b]
        p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
        comp.append(dict(stain=stain, a=a, b=b, median_diff=d.median(), a_better=int((d > 0).sum()), n=len(d), p=p))
    comp = pd.DataFrame(comp); comp['p_holm'] = holm(comp.p)
    return per_patient, pd.DataFrame(null_rows), comp, pd.concat(prot_rows)


# ---------------------------------------------------------------- C. collapse
def collapse(out_dir):
    """mutual correlation of CD4/CD8/CD20 (virtual vs MIST) and what each virtual channel tracks"""
    rows, track = [], {}
    for s, D in DATA.items():
        for pid, d in D.items():
            for src in ('gigatime', 'flash', 'mist'):
                if src == 'mist':
                    M = d['gt'][:, [d['proteins'].index(x) for x in ('CD4', 'CD8a', 'CD20')]]
                else:
                    M = d[src][:, [d['channels'].index(x) for x in ('CD4', 'CD8', 'CD20')]]
                R = spearmanr(M).correlation
                rows.append(dict(stain=s, patient=pid, source=src, mutual_rho=np.mean(R[np.triu_indices(3, 1)])))
        for m in ('gigatime', 'flash'):
            for ch in ('DAPI', 'CD4', 'CD8', 'CD20', 'CD68', 'CD3'):
                r = np.array([[sp(d[m][:, d['channels'].index(ch)], d['gt'][:, j]) for j in range(21)]
                              for d in D.values()])
                track[(s, m, ch)] = np.nanmedian(r, 0)
        prots = list(D.values())[0]['proteins']
    mut = pd.DataFrame(rows)
    mut.drop(columns='patient').to_csv(os.path.join(out_dir, 'collapse_mutual.csv'), index=False)
    stats = []
    for s in DATA:
        w = mut[mut.stain == s].pivot(index='patient', columns='source', values='mutual_rho')
        for m in ('gigatime', 'flash'):
            stats.append(dict(stain=s, model=m, virtual_median=w[m].median(), mist_median=w['mist'].median(),
                              virtual_higher=int((w[m] > w['mist']).sum()), n=len(w),
                              p=wilcoxon(w[m] - w['mist']).pvalue))
    pd.DataFrame(stats).to_csv(os.path.join(out_dir, 'collapse_mutual_stats.csv'), index=False)
    for s in DATA:
        t = pd.DataFrame({f'{m}:{ch}': v for (st, m, ch), v in track.items() if st == s}, index=prots).T
        t.round(3).to_csv(os.path.join(out_dir, f'collapse_tracking_{s}.csv'))
    return pd.DataFrame(stats)


def main():
    global IN, SAME, DATA
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='results/models', help='output folder of 02_run_models.py')
    ap.add_argument('--data', default='data', help='for same_section.txt')
    ap.add_argument('--out', default='results')
    ap.add_argument('--pas', default='PAS', help='stain name of the same-section slides')
    ap.add_argument('--he', default='HE', help='stain name of the other-section slides')
    a = ap.parse_args()
    IN = a.models
    f = os.path.join(a.data, 'same_section.txt')
    SAME = {l.strip() for l in open(f) if l.strip()}
    os.makedirs(a.out, exist_ok=True)
    DATA = {'PAS': {k: v for k, v in load(a.pas).items() if k in SAME}, 'HE': load(a.he)}
    for s, D in DATA.items():
        print(s, len(D), 'patients,', sum(len(d['gt']) for d in D.values()), 'squares', flush=True)
    out = lambda n: os.path.join(a.out, n)
    zs = pd.concat([direct(D, s) for s, D in DATA.items()])
    zs.to_csv(out('A_direct_per_patient.csv'), index=False)
    zrows = []
    for key, g in zs.groupby(['stain', 'model', 'channel', 'protein', 'kind', 'summary']):
        d = g.rho - g.rho_shift640
        zrows.append(dict(zip(['stain', 'model', 'channel', 'protein', 'kind', 'summary'], key), n_patients=len(g),
                          rho_median=g.rho.median(), shift640_median=g.rho_shift640.median(),
                          true_beats_shift=int((d > 0).sum()),
                          p_true_vs_shift=wilcoxon(d.dropna()).pvalue if d.notna().sum() > 5 else np.nan))
    zsum = pd.DataFrame(zrows); zsum['p_holm'] = holm(zsum.p_true_vs_shift.fillna(1))
    zsum.to_csv(out('A_direct_summary.csv'), index=False)
    direct_cells(a.pas).to_csv(out('A_direct_cells_PAS.csv'), index=False)
    summ, comp, prot = [], [], []
    for s, D in DATA.items():
        pp, sm, cp, pr = trained(D, s)
        pp.to_csv(out(f'B_trained_per_patient_{s}.csv')); summ.append(sm); comp.append(cp); prot.append(pr)
    pd.concat(summ).to_csv(out('B_trained_summary.csv'), index=False)
    pd.concat(comp).to_csv(out('B_model_comparisons.csv'), index=False)
    pd.concat(prot).groupby(['stain', 'features', 'protein']).rho.median().unstack(1).round(3).to_csv(
        out('B_per_protein.csv'))
    print(collapse(a.out).round(4).to_string(index=False))


if __name__ == '__main__':
    main()
