"""Analytic H(comma|12-TET) bound, model-entropy comparison, note-level conflation regression, and perde identification for high-ΔIC degrees; also derives the analytic jitter (uniform-choice) entropy bound."""

import csv, sqlite3
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from scipy import stats as ss

BASE       = Path("/Users/ugurozalp/makam_beklenti")          # local IDyOM output location; edit for your setup
SYMBTR_DIR = Path("/Users/ugurozalp/Downloads/SymbTr-master/txt")  # local SymbTr checkout; edit for your setup
OUTDIR     = BASE / "data" / "idyom_output"
RESULTS    = Path(__file__).resolve().parent.parent / "results"

MAKAMS = ["ussak", "huseyni", "nihavent"]
LABELS = {"ussak": "Uşşak", "huseyni": "Hüseyni", "nihavent": "Nihâvend"}
DATASET = {
    "ussak":    {"koma": 200, "tet12": 201},
    "huseyni":  {"koma": 202, "tet12": 203},
    "nihavent": {"koma": 204, "tet12": 205},
}
N_MIN = 50  # minimum notes per degree for 1D

# ── shared helpers ─────────────────────────────────────────────────────────────

def _int(s):
    try: return int(s)
    except: return None

def _float(s):
    try:
        v = float(s)
        return v if not (v != v) else None   # reject NaN
    except: return None

def read_dat_full(p):
    rows = []
    with open(p) as f:
        hdr = f.readline().split()
        for line in f:
            pts = line.split()
            row = dict(zip(hdr, pts))
            if row.get("cpitch.ic", "NA") == "NA": continue
            rec = {
                "mid":     int(row["melody.id"]),
                "nid":     int(row["note.id"]),
                "cp":      int(row["cpitch"]),
                "ic":      float(row["cpitch.ic"]),
                "entropy": _float(row.get("cpitch.entropy", "NA")),
                "name":    row.get("melody.name", "").strip('"'),
            }
            rows.append(rec)
    return rows

def get_karar(makam):
    db = OUTDIR / makam / "koma" / "idyom.db"
    with sqlite3.connect(db) as c:
        rows = c.execute(
            "SELECT COMPOSITION_ID, CPITCH FROM mtp_event "
            "WHERE DATASET_ID=? ORDER BY COMPOSITION_ID, ONSET",
            (DATASET[makam]["koma"],)).fetchall()
    last = {}
    for cid, cp in rows: last[cid] = cp
    return last

def build_note_records(makam):
    """Note-level records: delta_ic, entropy_koma/tet, deg, mid."""
    dat_k = sorted((OUTDIR/makam/"koma").glob("*.dat"))[0]
    dat_t = sorted((OUTDIR/makam/"tet12").glob("*.dat"))[0]
    karar = get_karar(makam)
    rk    = read_dat_full(dat_k)
    rt    = read_dat_full(dat_t)
    idx_t = {(r["mid"], r["nid"]): r for r in rt}
    recs  = []
    for r in rk:
        rt_row = idx_t.get((r["mid"], r["nid"]))
        if rt_row is None: continue
        k = karar.get(r["mid"] - 1)
        if k is None: continue
        cp_tet = round(r["cp"] * 12 / 53)
        recs.append({
            "mid":          r["mid"],
            "name":         r["name"],
            "cp_koma":      r["cp"],
            "cp_tet":       cp_tet,
            "ic_koma":      r["ic"],
            "ic_tet":       rt_row["ic"],
            "delta_ic":     r["ic"] - rt_row["ic"],
            "entropy_koma": r["entropy"],
            "entropy_tet":  rt_row["entropy"],
            "deg":          (r["cp"] - k) % 53,
            "makam":        makam,
        })
    return recs

def parse_txt_names(path):
    """Returns [(koma53, nota53_name)] for all pitched notes in a SymbTr TXT."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            pts = line.rstrip("\n").split("\t")
            if len(pts) < 5 or _int(pts[1]) != 9: continue
            koma = _int(pts[4])
            nota = pts[2].strip()
            if koma and koma > 0 and nota: rows.append((koma, nota))
    return rows

def get_koma_names(makam):
    """Build {koma53: set(nota53_names)} from all SymbTr TXT files for makam."""
    mapping = defaultdict(set)
    for txt in SYMBTR_DIR.glob(f"{makam}--*.txt"):
        for k, n in parse_txt_names(txt):
            mapping[k].add(n)
    return mapping

def h_bits(counts):
    """Shannon entropy in bits from Counter or dict of counts."""
    total = sum(counts.values())
    if total == 0: return 0.0
    return -sum((c/total)*np.log2(c/total) for c in counts.values() if c > 0)

def cohens_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na-1)*np.var(a,ddof=1)+(nb-1)*np.var(b,ddof=1))/(na+nb-2))
    return (np.mean(a)-np.mean(b))/sp if sp > 0 else 0.0

def fmt_p(p):
    return "< .001" if p < .001 else f"= {p:.4f}"

def piece_means(recs, field):
    per = defaultdict(list)
    for r in recs: per[r["mid"]].append(r[field])
    return {mid: np.mean(v) for mid, v in per.items()}

# ── OLS with small-sample-corrected cluster-robust SEs ───────────────────────

def ols_clustered(y, X, clusters):
    """OLS sandwich estimator with Stata-style small-sample correction."""
    y, X, cl = np.asarray(y, float), np.asarray(X, float), np.asarray(clusters)
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta    = XtX_inv @ (X.T @ y)
    resid   = y - X @ beta
    G       = len(np.unique(cl))
    meat    = np.zeros((k, k))
    for g in np.unique(cl):
        mask = cl == g
        e_g  = resid[mask]; X_g = X[mask]
        meat += X_g.T @ np.outer(e_g, e_g) @ X_g
    corr = (n/(n-k)) * (G/(G-1))
    V    = corr * XtX_inv @ meat @ XtX_inv
    se   = np.sqrt(np.diag(V).clip(0))
    df   = G - 1
    t    = beta / np.where(se > 0, se, np.inf)
    p    = 2 * ss.t.sf(np.abs(t), df)
    ss_res = float(resid @ resid)
    ss_tot = float(np.sum((y - y.mean())**2))
    r2     = 1 - ss_res/ss_tot if ss_tot > 0 else 0.0
    return beta, se, t, p, r2, df

# ═════════════════════════════════════════════════════════════════════════════
# Data loading (single pass, shared across all analyses)
# ═════════════════════════════════════════════════════════════════════════════

print("Loading note records and pitch name maps …")
all_recs   = {m: build_note_records(m) for m in MAKAMS}
koma_names = {m: get_koma_names(m) for m in MAKAMS}
for m in MAKAMS:
    print(f"  {LABELS[m]:12s}  n_notes={len(all_recs[m])}  "
          f"distinct_koma53={len(koma_names[m])}")

csv_rows = []   # accumulated across all sections

# ═════════════════════════════════════════════════════════════════════════════
# 1A — Analytic baseline  ΔH_baseline = H(comma | 12-TET)
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*72)
print("1A — Analytic Baseline  ΔH_baseline = H(comma | 12-TET class)")
print("═"*72)

def corpus_dh_baseline(recs):
    """Corpus-level H(comma | 12TET); returns (dh, {tet: {n,p_c,h_cond,n_koma}})."""
    tet_koma = defaultdict(Counter)
    for r in recs: tet_koma[r["cp_tet"]][r["cp_koma"]] += 1
    N = sum(sum(c.values()) for c in tet_koma.values())
    dh = 0.0
    bd = {}
    for tet, cnt in tet_koma.items():
        n_c = sum(cnt.values())
        p_c = n_c / N
        h_c = h_bits(cnt)
        dh += p_c * h_c
        bd[tet] = {"n": n_c, "p_c": p_c, "h_cond": h_c, "n_koma": len(cnt)}
    return dh, bd

macro = {}
print(f"\n  {'Maqam':12s}  {'ΔIC (note-level)':>18}  {'ΔH_baseline':>14}  {'R = ΔIC/ΔH':>12}")
print("  " + "─"*62)
for m in MAKAMS:
    recs = all_recs[m]
    mean_dic = np.mean([r["delta_ic"] for r in recs])
    dh_base, bd = corpus_dh_baseline(recs)
    R = mean_dic / dh_base if dh_base > 0 else np.nan
    macro[m] = {"mean_dic": mean_dic, "dh_base": dh_base, "R": R, "bd": bd}
    print(f"  {LABELS[m]:12s}  {mean_dic:18.6f}  {dh_base:14.6f}  {R:12.4f}")
    for key, val in [("mean_delta_ic", mean_dic), ("dh_baseline", dh_base), ("R_corpus", R)]:
        csv_rows.append({"section":"1A","maqam":LABELS[m],"key":"corpus","metric":key,"value":round(val,6)})

print(f"\n  12-TET classes with ≥2 distinct comma members (H > 0):")
print(f"  {'Maqam':12s}  {'12-TET':>7}  {'n notes':>8}  {'n comma':>8}  {'H(k|c)':>8}  {'p(c)':>7}")
print("  " + "─"*58)
for m in MAKAMS:
    for tet, info in sorted(macro[m]["bd"].items()):
        if info["n_koma"] >= 2:
            print(f"  {LABELS[m]:12s}  {tet:7d}  {info['n']:8d}  {info['n_koma']:8d}  "
                  f"{info['h_cond']:8.4f}  {info['p_c']:7.4f}")
            csv_rows.append({"section":"1A","maqam":LABELS[m],"key":f"tet{tet}",
                             "metric":"h_cond","value":round(info["h_cond"],6)})

# Piece-level R for statistical tests  (R_piece = piece_mean_ΔIC / corpus_ΔH_baseline)
R_pieces = {}
for m in MAKAMS:
    dh = macro[m]["dh_base"]
    pm = piece_means(all_recs[m], "delta_ic")
    R_pieces[m] = np.array([v/dh for v in pm.values() if dh > 0])

print(f"\n  Piece-level R  (R_piece = piece_mean_ΔIC / corpus_ΔH_baseline)")
print(f"  {'Maqam':12s}  {'n':>4}  {'mean R':>8}  {'median R':>10}")
print("  " + "─"*38)
for m in MAKAMS:
    print(f"  {LABELS[m]:12s}  {len(R_pieces[m]):4d}  {np.mean(R_pieces[m]):8.4f}  {np.median(R_pieces[m]):10.4f}")

H_kw, p_kw = ss.kruskal(*[R_pieces[m] for m in MAKAMS])
micro_R = np.concatenate([R_pieces["ussak"], R_pieces["huseyni"]])
ctrl_R  = R_pieces["nihavent"]
U_mw, p_mw = ss.mannwhitneyu(micro_R, ctrl_R, alternative="two-sided")
d_mw = cohens_d(micro_R, ctrl_R)
print(f"\n  Kruskal–Wallis (3 maqams): H = {H_kw:.4f},  p {fmt_p(p_kw)}")
print(f"  Mann–Whitney   (micro vs Nihâvend): U = {U_mw:.1f},  p {fmt_p(p_mw)},  d = {d_mw:+.4f}")
for metric, val in [("KW_H",H_kw),("KW_p",p_kw),("MW_U",U_mw),("MW_p",p_mw),("MW_d",d_mw)]:
    csv_rows.append({"section":"1A","maqam":"all","key":"stats","metric":metric,"value":round(val,6)})

# ═════════════════════════════════════════════════════════════════════════════
# 1B — Model entropy  ΔH_model = H_comma − H_12TET
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*72)
print("1B — Model Entropy  ΔH_model = H_comma − H_12TET")
print("═"*72)

print(f"\n  {'Maqam':12s}  {'H_comma':>10}  {'H_12TET':>10}  {'ΔH_model':>10}")
print("  " + "─"*48)

for m in MAKAMS:
    recs = all_recs[m]
    ek = [r["entropy_koma"] for r in recs if r["entropy_koma"] is not None]
    et = [r["entropy_tet"]  for r in recs if r["entropy_tet"]  is not None]
    dh = [r["entropy_koma"] - r["entropy_tet"] for r in recs
          if r["entropy_koma"] is not None and r["entropy_tet"] is not None]
    mhk = np.mean(ek); mht = np.mean(et); mdh = np.mean(dh)
    print(f"  {LABELS[m]:12s}  {mhk:10.5f}  {mht:10.5f}  {mdh:10.5f}")
    for metric, val in [("mean_H_koma",mhk),("mean_H_12tet",mht),("mean_dH_model",mdh)]:
        csv_rows.append({"section":"1B","maqam":LABELS[m],"key":"corpus","metric":metric,"value":round(val,6)})

# Piece-level Spearman(ΔIC, ΔH_model)
print(f"\n  Piece-level Spearman(ΔIC, ΔH_model):")
print(f"  {'Maqam':12s}  {'n':>4}  {'ρ':>8}  p")
print("  " + "─"*40)
for m in MAKAMS:
    recs = all_recs[m]
    per_mid_dic = defaultdict(list); per_mid_dh = defaultdict(list)
    for r in recs:
        per_mid_dic[r["mid"]].append(r["delta_ic"])
        if r["entropy_koma"] is not None and r["entropy_tet"] is not None:
            per_mid_dh[r["mid"]].append(r["entropy_koma"] - r["entropy_tet"])
    common = [mid for mid in per_mid_dic if mid in per_mid_dh]
    dic_arr = np.array([np.mean(per_mid_dic[mid]) for mid in common])
    dh_arr  = np.array([np.mean(per_mid_dh[mid])  for mid in common])
    rho, p_sp = ss.spearmanr(dic_arr, dh_arr)
    p_label = "p < .001" if p_sp < .001 else f"p = {p_sp:.3f}"
    print(f"  {LABELS[m]:12s}  {len(common):4d}  {rho:8.4f}  {p_label}")
    csv_rows.append({"section":"1B","maqam":LABELS[m],"key":"piece","metric":"spearman_rho","value":round(rho,4)})
    csv_rows.append({"section":"1B","maqam":LABELS[m],"key":"piece","metric":"spearman_p","value":round(p_sp,6)})

# ═════════════════════════════════════════════════════════════════════════════
# 1C — Note-level conflation regression
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*72)
print("1C — Note-level Conflation Regression")
print("  ΔIC ~ β₀ + β₁ · log₂(max_class_freq / this_note_freq)")
print("  Only 12-TET classes with ≥2 distinct comma members; piece-clustered SEs")
print("═"*72)

def make_conflation_arrays(recs, makam_offset=0):
    """
    Returns (delta_ic, log_ratio, piece_uid) for notes in multi-comma 12-TET classes.
    makam_offset ensures piece UIDs are globally unique when pooling maqams.
    """
    tet_koma_cnt = defaultdict(Counter)
    for r in recs: tet_koma_cnt[r["cp_tet"]][r["cp_koma"]] += 1

    multi_tet = {t for t, c in tet_koma_cnt.items() if len(c) >= 2}

    tet_koma_freq = {}
    tet_max_freq  = {}
    for tet, cnt in tet_koma_cnt.items():
        if tet not in multi_tet: continue
        n_c = sum(cnt.values())
        freqs = {k: v/n_c for k, v in cnt.items()}
        tet_koma_freq[tet] = freqs
        tet_max_freq[tet]  = max(freqs.values())

    dic_list, lr_list, mid_list = [], [], []
    for r in recs:
        tet = r["cp_tet"]
        if tet not in multi_tet: continue
        f_this = tet_koma_freq[tet].get(r["cp_koma"])
        if not f_this or f_this <= 0: continue
        lr = np.log2(tet_max_freq[tet] / f_this)
        dic_list.append(r["delta_ic"])
        lr_list.append(lr)
        mid_list.append(makam_offset + r["mid"])

    return (np.array(dic_list), np.array(lr_list),
            np.array(mid_list), len(multi_tet))

print(f"\n  {'Maqam':12s}  {'n notes':>8}  {'n classes':>10}  "
      f"{'slope':>8}  {'SE':>7}  {'95% CI':>20}  p          {'R²':>7}")
print("  " + "─"*90)

def report_reg(label, dic_arr, lr_arr, mid_arr, n_classes):
    if len(lr_arr) < 20:
        print(f"  {label:12s}  insufficient data"); return None
    X    = np.column_stack([np.ones(len(lr_arr)), lr_arr])
    beta, se, t, p, r2, df = ols_clustered(dic_arr, X, mid_arr)
    slope, se_s = beta[1], se[1]
    t_crit = ss.t.ppf(0.975, df)
    ci_lo, ci_hi = slope - t_crit*se_s, slope + t_crit*se_s
    p_s = p[1]
    n_cl = len(np.unique(mid_arr))
    p_str = "< .001" if p_s < .001 else f"= {p_s:.4f}"
    print(f"  {label:12s}  {len(lr_arr):8d}  {n_classes:10d}  "
          f"{slope:8.4f}  {se_s:7.4f}  [{ci_lo:+7.4f},{ci_hi:+7.4f}]  p {p_str:>8}  {r2:7.4f}")
    return dict(slope=slope, se=se_s, ci_lo=ci_lo, ci_hi=ci_hi, p=p_s,
                r2=r2, n=len(lr_arr), n_classes=n_classes, n_clusters=n_cl)

reg_results = {}
for m in MAKAMS:
    dic_arr, lr_arr, mid_arr, n_cl = make_conflation_arrays(all_recs[m])
    res = report_reg(LABELS[m], dic_arr, lr_arr, mid_arr, n_cl)
    reg_results[m] = res
    if res:
        for metric, val in res.items():
            csv_rows.append({"section":"1C","maqam":LABELS[m],"key":"OLS","metric":metric,"value":round(float(val),6)})

# Combined (pool maqams, offset piece IDs to prevent collisions)
print()
offsets = {"ussak": 0, "huseyni": 10000, "nihavent": 20000}
pool_dic, pool_lr, pool_mid = [], [], []
pool_n_cl = 0
for m in MAKAMS:
    d, lr, mid, n_cl = make_conflation_arrays(all_recs[m], makam_offset=offsets[m])
    pool_dic.extend(d); pool_lr.extend(lr); pool_mid.extend(mid)
    pool_n_cl += n_cl
pool_dic = np.array(pool_dic); pool_lr = np.array(pool_lr); pool_mid = np.array(pool_mid)
res_comb = report_reg("Combined", pool_dic, pool_lr, pool_mid, pool_n_cl)
if res_comb:
    for metric, val in res_comb.items():
        csv_rows.append({"section":"1C","maqam":"combined","key":"OLS","metric":metric,"value":round(float(val),6)})

# ═════════════════════════════════════════════════════════════════════════════
# 1D — High-ΔIC degree pitch names
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*72)
print(f"1D — High-ΔIC Degrees  (n ≥ {N_MIN}, sorted by mean ΔIC ↓)")
print("═"*72)

for m in MAKAMS:
    recs   = all_recs[m]
    names  = koma_names[m]

    # 12-TET → set of all koma values in this maqam
    tet_to_komas = defaultdict(set)
    for r in recs: tet_to_komas[r["cp_tet"]].add(r["cp_koma"])

    # Group by degree
    deg_recs = defaultdict(list)
    for r in recs: deg_recs[r["deg"]].append(r)

    qualified = [(d, recs_d) for d, recs_d in deg_recs.items() if len(recs_d) >= N_MIN]
    qualified.sort(key=lambda x: np.mean([r["delta_ic"] for r in x[1]]), reverse=True)

    print(f"\n  {LABELS[m]}")
    print(f"  {'d':>4}  {'n':>6}  {'ΔIC':>8}  {'Koma53':>14}  {'Nota53':>28}  {'12-TET':>6}  companions")
    print("  " + "─"*100)

    for d, recs_d in qualified:
        n    = len(recs_d)
        mdc  = np.mean([r["delta_ic"] for r in recs_d])
        komas_at_d = sorted({r["cp_koma"] for r in recs_d})
        nota_set   = set()
        for k in komas_at_d: nota_set.update(names.get(k, {f"K{k}"}))
        tet_vals   = sorted({round(k*12/53) for k in komas_at_d})
        companions = set()
        for tet in tet_vals:
            for k in tet_to_komas.get(tet, set()):
                if k not in komas_at_d:
                    companions.update(names.get(k, {f"K{k}"}))

        koma_str = ",".join(str(k) for k in komas_at_d)
        nota_str = ",".join(sorted(nota_set))
        tet_str  = ",".join(str(t) for t in tet_vals)
        comp_str = ",".join(sorted(companions)) or "—"

        print(f"  {d:4d}  {n:6d}  {mdc:+8.4f}  {koma_str:>14}  {nota_str:>28}  {tet_str:>6}  {comp_str}")

        for metric, val in [("n_notes",n),("mean_delta_ic",round(mdc,4)),
                             ("koma53",koma_str),("nota53",nota_str),
                             ("tet12_class",tet_str),("companions",comp_str)]:
            csv_rows.append({"section":"1D","maqam":LABELS[m],
                             "key":f"d={d}","metric":metric,"value":str(val)})

    # Always show Nihâvend d=17 and d=36 even if n < N_MIN
    if m == "nihavent":
        specials = [(d, deg_recs[d]) for d in [17, 36] if d in deg_recs]
        for d, recs_d in specials:
            n = len(recs_d); mdc = np.mean([r["delta_ic"] for r in recs_d])
            komas_at_d = sorted({r["cp_koma"] for r in recs_d})
            nota_set   = set()
            for k in komas_at_d: nota_set.update(names.get(k, {f"K{k}"}))
            tet_vals   = sorted({round(k*12/53) for k in komas_at_d})
            companions = set()
            for tet in tet_vals:
                for k in tet_to_komas.get(tet, set()):
                    if k not in komas_at_d:
                        companions.update(names.get(k, {f"K{k}"}))
            koma_str = ",".join(str(k) for k in komas_at_d)
            nota_str = ",".join(sorted(nota_set))
            tet_str  = ",".join(str(t) for t in tet_vals)
            comp_str = ",".join(sorted(companions)) or "—"
            flag = f"  {'(n<50)' if n<N_MIN else ''}"
            print(f"  d={d:2d}  n={n:4d}  ΔIC={mdc:+.4f}  koma={koma_str}  "
                  f"nota={nota_str}  12TET={tet_str}  companions={comp_str}{flag}")

# ═════════════════════════════════════════════════════════════════════════════
# 1E — Analytic jitter bound: E_uniform[H(comma|12TET)] = Σ_c p(c)·log2(k_c)
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*72)
print("1E — Analytic Jitter Bound (uniform-choice entropy, no IDyOM re-run)")
print("  E_uniform[H] = Σ_c p(c)·log2(k_c), k_c = #distinct komas in class c")
print("═"*72)

jitter_rows = []
for m in MAKAMS:
    recs = all_recs[m]
    tet_koma = defaultdict(Counter)
    for r in recs: tet_koma[r["cp_tet"]][r["cp_koma"]] += 1
    N = sum(sum(c.values()) for c in tet_koma.values())
    e_unif = sum((sum(c.values())/N) * np.log2(len(c)) for c in tet_koma.values())
    obs_mean = float(np.mean([r["delta_ic"] for r in recs]))
    dh_base  = macro[m]["dh_base"]
    print(f"  {LABELS[m]:12s}  E_uniform[H] = {e_unif:.5f}   "
          f"(empirical ΔH_baseline = {dh_base:.5f}, ratio {e_unif/dh_base:.2f}x)")
    for key, val, status in [
        ("obs_note_mean_dic", obs_mean, "valid"),
        ("E_uniform_H_analytic", e_unif, "valid"),
        ("empirical_dH_baseline_1A", dh_base, "valid"),
    ]:
        jitter_rows.append({"maqam": LABELS[m], "key": key, "value": round(val, 6), "status": status})

out_jitter = RESULTS / "jitter_control.csv"
with open(out_jitter, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["maqam","key","value","status"])
    w.writeheader(); w.writerows(jitter_rows)
print(f"\n  → Saved: {out_jitter}")

# ═════════════════════════════════════════════════════════════════════════════
# Save main CSV
# ═════════════════════════════════════════════════════════════════════════════

out = RESULTS / "baselines_and_diagnostics.csv"
fields = ["section", "maqam", "key", "metric", "value"]
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for row in csv_rows:
        w.writerow({k: row.get(k, "") for k in fields})
print(f"\n\n  → Saved: {out}  ({len(csv_rows)} rows)")
