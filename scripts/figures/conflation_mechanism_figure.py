"""Publication figure: note-level ΔIC vs. within-class frequency disparity, showing the conflation mechanism behind ΔIC."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from scipy import stats as ss

BASE   = Path("/Users/ugurozalp/makam_beklenti")   # local IDyOM output location; edit for your setup
OUTDIR = BASE / "data" / "idyom_output"
FIGOUT = Path(__file__).resolve().parent.parent.parent / "figures" / "Figure_mechanism.png"
FIGOUT.parent.mkdir(exist_ok=True)

MAKAMS = ["ussak", "huseyni", "nihavent"]
LABELS = {"ussak": "Uşşak", "huseyni": "Hüseyni", "nihavent": "Nihâvend"}

BIN_WIDTH = 0.5

# ── data loading ────────────────────────────────────────────────────────────

def koma_to_tet(k): return round(int(k) * 12 / 53)

def read_dat_cp_ic(path):
    with open(path) as f: hdr = f.readline().split()
    mid_i, nid_i = hdr.index("melody.id"), hdr.index("note.id")
    cp_i, ic_i   = hdr.index("cpitch"), hdr.index("cpitch.ic")
    out = {}
    with open(path) as f:
        f.readline()
        for line in f:
            pts = line.split()
            if pts[ic_i] == "NA": continue
            out[(int(pts[mid_i]), int(pts[nid_i]))] = (int(pts[cp_i]), float(pts[ic_i]))
    return out

def get_dat(dir_path):
    return sorted(dir_path.glob("*.dat"))[0]

def load_recs(makam):
    dk = read_dat_cp_ic(get_dat(OUTDIR / makam / "koma"))
    dt = read_dat_cp_ic(get_dat(OUTDIR / makam / "tet12"))
    common = sorted(set(dk) & set(dt))
    recs = []
    for (mid, nid) in common:
        cp, ic_k = dk[(mid, nid)]
        _, ic_t  = dt[(mid, nid)]
        recs.append({"mid": mid, "cp": cp, "cp_tet": koma_to_tet(cp),
                     "delta_ic": ic_k - ic_t})
    return recs

def make_conflation_arrays(recs):
    tet_koma_cnt = defaultdict(Counter)
    for r in recs: tet_koma_cnt[r["cp_tet"]][r["cp"]] += 1
    multi_tet = {t for t, c in tet_koma_cnt.items() if len(c) >= 2}

    tet_koma_freq, tet_max_freq = {}, {}
    for tet, cnt in tet_koma_cnt.items():
        if tet not in multi_tet: continue
        n_c = sum(cnt.values())
        freqs = {k: v / n_c for k, v in cnt.items()}
        tet_koma_freq[tet] = freqs
        tet_max_freq[tet] = max(freqs.values())

    dic_list, lr_list, mid_list = [], [], []
    for r in recs:
        tet = r["cp_tet"]
        if tet not in multi_tet: continue
        f_this = tet_koma_freq[tet].get(r["cp"])
        if not f_this or f_this <= 0: continue
        lr = np.log2(tet_max_freq[tet] / f_this)
        dic_list.append(r["delta_ic"]); lr_list.append(lr); mid_list.append(r["mid"])
    return np.array(dic_list), np.array(lr_list), np.array(mid_list)

# ── clustered OLS (same as 1C) ────────────────────────────────────────────────

def ols_clustered(y, X, clusters):
    y, X, cl = np.asarray(y, float), np.asarray(X, float), np.asarray(clusters)
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    G = len(np.unique(cl))
    meat = np.zeros((k, k))
    for g in np.unique(cl):
        mask = cl == g
        e_g, X_g = resid[mask], X[mask]
        meat += X_g.T @ np.outer(e_g, e_g) @ X_g
    corr = (n / (n - k)) * (G / (G - 1))
    V = corr * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(V).clip(0))
    df = G - 1
    t = beta / np.where(se > 0, se, np.inf)
    p = 2 * ss.t.sf(np.abs(t), df)
    ss_res = float(resid @ resid)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return beta, se, t, p, r2, df

# ── binning with piece-clustered 95% CI ───────────────────────────────────────

def binned_stats(dic, lr, mid):
    lo = np.floor(lr.min() / BIN_WIDTH) * BIN_WIDTH
    hi = np.ceil(lr.max() / BIN_WIDTH) * BIN_WIDTH
    edges = np.arange(lo, hi + BIN_WIDTH, BIN_WIDTH)
    rows = []
    for i in range(len(edges) - 1):
        e0, e1 = edges[i], edges[i + 1]
        mask = (lr >= e0) & (lr < e1) if i < len(edges) - 2 else (lr >= e0) & (lr <= e1)
        n_notes = mask.sum()
        if n_notes == 0:
            continue
        bin_mid, bin_dic, bin_pieces = mid[mask], dic[mask], mid[mask]
        per_piece = defaultdict(list)
        for m, v in zip(bin_mid, bin_dic): per_piece[m].append(v)
        piece_means = np.array([np.mean(v) for v in per_piece.values()])
        G = len(piece_means)
        bin_mean = piece_means.mean()
        if G >= 2:
            se = piece_means.std(ddof=1) / np.sqrt(G)
            tcrit = ss.t.ppf(0.975, G - 1)
            ci_lo, ci_hi = bin_mean - tcrit * se, bin_mean + tcrit * se
        else:
            se, ci_lo, ci_hi = np.nan, np.nan, np.nan
        rows.append({"x_lo": e0, "x_hi": e1, "x_mid": (e0 + e1) / 2,
                     "n_notes": int(n_notes), "n_pieces": G,
                     "mean_dic": bin_mean, "se": se, "ci_lo": ci_lo, "ci_hi": ci_hi})
    return rows

# ── main ───────────────────────────────────────────────────────────────────

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 9,
    'axes.titlesize': 10.5,
    'axes.labelsize': 10,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'figure.dpi': 300,
})

fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), dpi=300, sharey=True)
YLIM = (-2, 8)   # meaningful range where the relationship is visible;
                 # sparse extreme bins (n_pieces often <2, huge CIs) are
                 # excluded from the scatter display -- reported below

print("="*90)
print("Figure_mechanism — conflation-disparity data (note-level)")
print("="*90)

all_bin_tables = {}

for ax, m in zip(axes, MAKAMS):
    recs = load_recs(m)
    dic, lr, mid = make_conflation_arrays(recs)

    X = np.column_stack([np.ones(len(lr)), lr])
    beta, se, t, p, r2, df = ols_clustered(dic, X, mid)
    slope, se_s, p_s = beta[1], se[1], p[1]

    bins = binned_stats(dic, lr, mid)
    all_bin_tables[m] = bins

    print(f"\n  {LABELS[m]}")
    print(f"    x range: [{lr.min():.3f}, {lr.max():.3f}]   n_notes={len(lr)}   "
          f"n_bins={len(bins)}")
    print(f"    {'bin':>14}  {'n_notes':>8}  {'n_pieces':>9}  {'mean ΔIC':>10}  "
          f"{'95% CI':>20}")
    for b in bins:
        ci_str = f"[{b['ci_lo']:+.4f},{b['ci_hi']:+.4f}]" if b['ci_lo']==b['ci_lo'] else "  (n_pieces<2)"
        print(f"    [{b['x_lo']:+.1f},{b['x_hi']:+.1f})  {b['n_notes']:8d}  {b['n_pieces']:9d}  "
              f"{b['mean_dic']:+10.4f}  {ci_str:>20}")
    print(f"    OLS (note-level, piece-clustered SE): "
          f"beta={slope:.4f}  SE={se_s:.4f}  p={'<.001' if p_s<.001 else f'{p_s:.4f}'}  R2={r2:.4f}")

    # split bins into in-range (plotted) vs out-of-range (excluded from the
    # scatter display, but still part of the underlying OLS fit above)
    in_range  = [b for b in bins if YLIM[0] <= b["mean_dic"] <= YLIM[1]]
    out_range = [b for b in bins if b not in in_range]
    if out_range:
        print(f"    CLIPPED FROM DISPLAY (mean ΔIC outside {YLIM}): "
              + ", ".join(f"[{b['x_lo']:+.1f},{b['x_hi']:+.1f})->{b['mean_dic']:+.2f} bits "
                          f"(n={b['n_notes']}, pieces={b['n_pieces']})" for b in out_range))

    xs = np.array([b["x_mid"] for b in in_range])
    ys = np.array([b["mean_dic"] for b in in_range])
    ns = np.array([b["n_notes"] for b in in_range])
    los = np.array([b["mean_dic"] - b["ci_lo"] if b["ci_lo"]==b["ci_lo"] else 0 for b in in_range])
    his = np.array([min(b["ci_hi"], YLIM[1]) - b["mean_dic"] if b["ci_hi"]==b["ci_hi"] else 0 for b in in_range])

    # log-scaled marker size: linear/sqrt scaling let the x=0 bin (>90% of
    # all notes in most maqams) visually swamp every other bin
    log_ns = np.log10(ns)
    sizes = 12 + 220 * (log_ns - log_ns.min()) / max(log_ns.max() - log_ns.min(), 1e-9)

    ax.errorbar(xs, ys, yerr=[los, his], fmt='none', ecolor='#555555',
                elinewidth=0.9, capsize=3, zorder=2)
    ax.scatter(xs, ys, s=sizes, color='#333333', alpha=0.85,
               edgecolors='white', linewidths=0.6, zorder=3)

    # regression line over observed x range (note-level fit) -- dark/dashed
    # so it stays visible in grayscale reproduction
    xr = np.linspace(lr.min(), lr.max(), 100)
    yr = beta[0] + beta[1] * xr
    ax.plot(xr, yr, color='#1a1a1a', lw=1.5, ls='--', zorder=1)

    ax.axhline(0, color='black', linestyle=':', linewidth=0.8, alpha=0.6, zorder=0)
    ax.set_ylim(*YLIM)

    p_str = "p < .001" if p_s < .001 else f"p = {p_s:.4f}"
    stat_text = f"β = {slope:.2f}\nSE = {se_s:.3f}\n{p_str}\nR² = {r2:.3f}"
    ax.text(0.05, 0.95, stat_text, transform=ax.transAxes, fontsize=8.3,
             va='top', ha='left',
             bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                        edgecolor='#999999', linewidth=0.7))

    ax.set_title(LABELS[m], fontsize=11, fontweight='bold')
    ax.set_xlabel(r"log$_2$(max class freq / this-note freq)")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[0].set_ylabel("ΔIC [bits], note-level")

fig.suptitle("Information gain scales with frequency disparity within conflated 12-TET classes",
             fontsize=13, fontweight='bold', y=1.04)

# NOTE: the in-figure caption text (marker-size / CI / clipping / regression-
# data explanation) was intentionally removed -- that information now belongs
# in the manuscript's figure caption, not rendered on the figure itself. The
# console output above still reports the clipped bins and their values.

plt.tight_layout()
plt.savefig(FIGOUT, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nSaved: {FIGOUT}")

# ── programmatic overlap / clipping check ─────────────────────────────────────

def validate_figure(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_bbox = fig.bbox
    boxes = []
    for ax in fig.axes:
        candidates = list(ax.texts) + [ax.title, ax.xaxis.label, ax.yaxis.label]
        candidates += list(ax.get_xticklabels()) + list(ax.get_yticklabels())
        for t in candidates:
            txt = t.get_text()
            if not txt: continue
            bb = t.get_window_extent(renderer=renderer)
            if bb.width == 0 or bb.height == 0 or not np.isfinite([bb.x0,bb.x1,bb.y0,bb.y1]).all():
                continue
            boxes.append((txt, bb))
    problems = []
    for txt, bb in boxes:
        if not (fig_bbox.x0 - 1 <= bb.x0 and bb.x1 <= fig_bbox.x1 + 1 and
                fig_bbox.y0 - 1 <= bb.y0 and bb.y1 <= fig_bbox.y1 + 1):
            problems.append(f"CLIPPED: '{txt}' extends outside figure bounds")
    n = len(boxes)
    for i in range(n):
        for j2 in range(i + 1, n):
            t1, b1 = boxes[i]; t2, b2 = boxes[j2]
            if t1 == t2: continue
            if b1.overlaps(b2):
                ix0, iy0 = max(b1.x0, b2.x0), max(b1.y0, b2.y0)
                ix1, iy1 = min(b1.x1, b2.x1), min(b1.y1, b2.y1)
                inter_area = max(0, ix1 - ix0) * max(0, iy1 - iy0)
                min_area = min(b1.width * b1.height, b2.width * b2.height)
                if min_area > 0 and inter_area / min_area > 0.15:
                    problems.append(f"OVERLAP: '{t1}' overlaps '{t2}' "
                                    f"({100*inter_area/min_area:.0f}% of smaller box)")
    print(f"\n--- Layout validation: {name} ---")
    if problems:
        for p in problems: print(f"  ISSUE: {p}")
    else:
        print("  (a) no clipped text   (b) no overlapping text boxes   "
              "(c) no out-of-bounds elements")
    return problems

validate_figure(fig, "Figure_mechanism.png")
