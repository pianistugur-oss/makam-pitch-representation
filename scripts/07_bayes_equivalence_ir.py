"""
Bayesian and equivalence analyses for null ΔIR result.

Groups:
  microtonal (Uşşak + Hüseyni combined)  vs  control (Nihâvend)

Analyses (both on rank-transformed and raw ΔIR):
  1. Bayes factor  — JZS prior r = 0.707, two-sided independent t-test
  2. TOST          — SESOI = ±d_ref = ±0.37 (from ΔIC Mann–Whitney), Welch t
"""

import csv, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats as ss
import pingouin as pg

BASE       = Path("/Users/ugurozalp/makam_beklenti")
SYMBTR_DIR = Path("/Users/ugurozalp/Downloads/SymbTr-master/txt")
OUTDIR     = BASE / "data" / "idyom_output"

MAKAMS  = ["ussak", "huseyni", "nihavent"]
LABELS  = {"ussak": "Uşşak", "huseyni": "Hüseyni", "nihavent": "Nihâvend"}
DATASET = {
    "ussak":    {"koma": 200, "tet12": 201},
    "huseyni":  {"koma": 202, "tet12": 203},
    "nihavent": {"koma": 204, "tet12": 205},
}

SESOI_D = 0.37   # Cohen's d from ΔIC Mann–Whitney (effect to match)
JZS_R   = 0.707  # default Cauchy scale factor

# ── IR computation (same as ir_analysis.py / figures_english.py) ─────────────

def _int(s):
    try: return int(s)
    except: return None

def _ir_score(I1, I2, large, small):
    a1, a2   = abs(I1), abs(I2)
    same_dir = (I1 > 0 and I2 > 0) or (I1 < 0 and I2 < 0)
    rdir = (1.0 if same_dir else 0.0) if a1 > large else \
           (0.0 if same_dir else 1.0) if a1 <= small else 0.0
    diff = 1.0 if a2 < a1 else 0.0
    rret = 1.0 if abs(I1 + I2) <= small else 0.0
    prox = max(0.0, 1.0 - a2 / large)
    clos = 1.0 if (not same_dir) and (a2 <= small) else 0.0
    return rdir + diff + rret + prox + clos

KOMA_LARGE, KOMA_SMALL = 27, 9
TET_LARGE,  TET_SMALL  = 6,  2

def _get_karar(makam):
    import sqlite3
    db = OUTDIR / makam / "koma" / "idyom.db"
    with sqlite3.connect(db) as c:
        rows = c.execute(
            "SELECT COMPOSITION_ID, CPITCH FROM mtp_event "
            "WHERE DATASET_ID=? ORDER BY COMPOSITION_ID, ONSET",
            (DATASET[makam]["koma"],),
        ).fetchall()
    last = {}
    for cid, cp in rows:
        last[cid] = cp
    return last

def _get_pieces(makam):
    import sqlite3
    db = OUTDIR / makam / "koma" / "idyom.db"
    with sqlite3.connect(db) as c:
        return c.execute(
            "SELECT COMPOSITION_ID, DESCRIPTION FROM mtp_composition "
            "WHERE DATASET_ID=? ORDER BY COMPOSITION_ID",
            (DATASET[makam]["koma"],),
        ).fetchall()

def _parse_txt(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            pts = line.rstrip("\n").split("\t")
            if len(pts) < 9: continue
            if _int(pts[1]) != 9: continue
            koma = _int(pts[4])
            if not koma or koma <= 0: continue
            rows.append((koma, round(koma * 12 / 53)))
    return rows

def load_delta_ir(makam):
    """Returns list of piece-level delta_ir values."""
    pieces    = _get_pieces(makam)
    karar_map = _get_karar(makam)
    piece_dirs = []
    for cid_db, name in pieces:
        txt = SYMBTR_DIR / (name + ".txt")
        if not txt.exists(): continue
        notes = _parse_txt(txt)
        if len(notes) < 3: continue
        karar = karar_map.get(cid_db)
        if karar is None: continue
        komas  = [n[0] for n in notes]
        tet12s = [n[1] for n in notes]
        sks, sts = [], []
        for i in range(len(notes) - 2):
            sks.append(_ir_score(komas[i+1]-komas[i],  komas[i+2]-komas[i+1],  KOMA_LARGE, KOMA_SMALL))
            sts.append(_ir_score(tet12s[i+1]-tet12s[i], tet12s[i+2]-tet12s[i+1], TET_LARGE,  TET_SMALL))
        if not sks: continue
        piece_dirs.append(np.mean(sks) - np.mean(sts))
    return np.array(piece_dirs)

# ── statistics helpers ────────────────────────────────────────────────────────

def pooled_sd(a, b):
    na, nb = len(a), len(b)
    return np.sqrt(
        ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1))
        / (na + nb - 2)
    )

def cohens_d(a, b):
    sp = pooled_sd(a, b)
    return (np.mean(a) - np.mean(b)) / sp if sp > 0 else 0.0

def run_analysis(a, b, label, sesoi_d=SESOI_D):
    """
    Returns dict with all stats for one data variant (ranked or raw).
    a = microtonal group, b = control group.
    """
    na, nb = len(a), len(b)

    # Welch t-test (reference)
    t_welch, p_welch = ss.ttest_ind(a, b, equal_var=False)
    df_welch = (np.var(a, ddof=1)/na + np.var(b, ddof=1)/nb)**2 / (
        (np.var(a, ddof=1)/na)**2/(na-1) + (np.var(b, ddof=1)/nb)**2/(nb-1)
    )
    d_obs = cohens_d(a, b)

    # 1. Bayes Factor
    bf10 = pg.bayesfactor_ttest(t_welch, na, nb, paired=False,
                                alternative="two-sided", r=JZS_R)
    bf01 = 1.0 / bf10

    # 2. TOST — Welch-based, two one-sided t-tests, bound = SESOI_D × pooled SD
    sp    = pooled_sd(a, b)
    bound = sesoi_d * sp          # convert Cohen's d to raw units
    na_f, nb_f = float(na), float(nb)
    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
    se    = np.sqrt(var_a / na_f + var_b / nb_f)
    delta = np.mean(a) - np.mean(b)
    # Welch df
    df_tost = (var_a/na_f + var_b/nb_f)**2 / (
        (var_a/na_f)**2 / (na_f - 1) + (var_b/nb_f)**2 / (nb_f - 1)
    )
    # H01: delta ≤ −bound  →  t_lo = (delta + bound)/SE; reject if t_lo > t_crit
    t_lower = (delta + bound) / se
    p_lower = float(ss.t.sf(t_lower, df_tost))   # upper-tail p
    # H02: delta ≥ +bound  →  t_up = (delta − bound)/SE; reject if t_up < −t_crit
    t_upper = (delta - bound) / se
    p_upper = float(ss.t.cdf(t_upper, df_tost))  # lower-tail p
    p_tost  = max(p_lower, p_upper)               # conservative TOST p

    # Interpretation flags
    bf_interp = ("null (moderate)"  if bf01 > 3 else
                 "null (anecdotal)" if bf01 > 1 else
                 "H1 (anecdotal)"   if bf10 > 1 else "inconclusive")
    tost_equiv = p_lower < 0.05 and p_upper < 0.05

    return {
        "label":      label,
        "n_micro":    na,
        "n_ctrl":     nb,
        "mean_micro": float(np.mean(a)),
        "mean_ctrl":  float(np.mean(b)),
        "d_obs":      float(d_obs),
        "t_welch":    float(t_welch),
        "df_welch":   float(df_welch),
        "p_welch":    float(p_welch),
        "bf10":       float(bf10),
        "bf01":       float(bf01),
        "bf_interp":  bf_interp,
        "sesoi_d":    sesoi_d,
        "bound_raw":  float(bound),
        "df_tost":    float(df_tost),
        "p_tost_lower": float(p_lower),
        "p_tost_upper": float(p_upper),
        "p_tost_max":   float(p_tost),
        "tost_equiv":   tost_equiv,
    }

def print_block(r, title):
    print(f"\n  {'─'*62}")
    print(f"  {title}")
    print(f"  {'─'*62}")
    print(f"  Groups : microtonal n={r['n_micro']}  vs  Nihâvend n={r['n_ctrl']}")
    print(f"  Means  : micro = {r['mean_micro']:+.4f}   ctrl = {r['mean_ctrl']:+.4f}")
    print(f"  Cohen's d (observed) = {r['d_obs']:+.4f}")
    print()
    print(f"  ① Bayes Factor  (JZS prior r = {JZS_R})")
    print(f"     Welch t({r['df_welch']:.1f}) = {r['t_welch']:.4f},  p = {r['p_welch']:.4f}")
    print(f"     BF₁₀ = {r['bf10']:.4f}   →   BF₀₁ = {r['bf01']:.4f}")
    print(f"     Interpretation: {r['bf_interp']}")
    print(f"     [BF₀₁ > 3 = moderate evidence for null; > 10 = strong]")
    print()
    print(f"  ② TOST Equivalence  (SESOI = ±{r['sesoi_d']} Cohen's d  →  bound = ±{r['bound_raw']:.4f} raw)")
    print(f"     df = {r['df_tost']:.1f}")
    print(f"     p (lower bound, H0: diff ≤ −bound) = {r['p_tost_lower']:.4f}")
    print(f"     p (upper bound, H0: diff ≥ +bound) = {r['p_tost_upper']:.4f}")
    print(f"     TOST p (max of two)               = {r['p_tost_max']:.4f}")
    tost_str = "✓  Both p < .05 → equivalence established" if r["tost_equiv"] \
               else "✗  At least one p ≥ .05 → equivalence not established"
    print(f"     {tost_str}")

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 66)
    print("Bayesian & Equivalence Analysis — Piece-level ΔIR")
    print("Microtonal group (Uşşak + Hüseyni)  vs  Nihâvend")
    print("=" * 66)

    # Load ΔIR
    print("\nLoading ΔIR data …")
    ir = {m: load_delta_ir(m) for m in MAKAMS}
    for m in MAKAMS:
        print(f"  {LABELS[m]:12s}  n = {len(ir[m])}  "
              f"mean = {np.mean(ir[m]):+.5f}  sd = {np.std(ir[m], ddof=1):.5f}")

    micro_raw = np.concatenate([ir["ussak"], ir["huseyni"]])
    ctrl_raw  = ir["nihavent"]

    # Rank transform (combined pool → ranks → split)
    all_vals  = np.concatenate([micro_raw, ctrl_raw])
    ranks_all = ss.rankdata(all_vals)
    n_micro   = len(micro_raw)
    micro_rank = ranks_all[:n_micro]
    ctrl_rank  = ranks_all[n_micro:]

    # --- Ranked ---
    res_rank = run_analysis(micro_rank, ctrl_rank, "rank-transformed ΔIR")
    print_block(res_rank, "RANKED DATA  (rank-transform then t-test / TOST)")

    # --- Raw ---
    res_raw = run_analysis(micro_raw, ctrl_raw, "raw ΔIR")
    print_block(res_raw, "RAW DATA  (sensitivity control)")

    # ── Interpretation summary ─────────────────────────────────────────────
    print(f"\n{'═'*66}")
    print("INTERPRETATION SUMMARY")
    print(f"{'═'*66}")
    print(f"""
  Thresholds used:
    BF₀₁ > 3   → moderate evidence for null hypothesis (no group diff)
    BF₀₁ > 10  → strong evidence for null
    TOST both p < .05 → difference falls within equivalence band (±{SESOI_D} d)

  Ranked:
    BF₀₁ = {res_rank['bf01']:.3f}  ({res_rank['bf_interp']})
    TOST  = {'equivalent' if res_rank['tost_equiv'] else 'not equivalent'}  (p_lower={res_rank['p_tost_lower']:.4f}, p_upper={res_rank['p_tost_upper']:.4f})

  Raw:
    BF₀₁ = {res_raw['bf01']:.3f}  ({res_raw['bf_interp']})
    TOST  = {'equivalent' if res_raw['tost_equiv'] else 'not equivalent'}  (p_lower={res_raw['p_tost_lower']:.4f}, p_upper={res_raw['p_tost_upper']:.4f})
""")

    # ── CSV ───────────────────────────────────────────────────────────────
    fields = [
        "label", "n_micro", "n_ctrl", "mean_micro", "mean_ctrl",
        "d_obs", "t_welch", "df_welch", "p_welch",
        "bf10", "bf01", "bf_interp",
        "sesoi_d", "bound_raw", "df_tost",
        "p_tost_lower", "p_tost_upper", "p_tost_max", "tost_equiv",
    ]
    out = BASE / "bayes_equivalence_ir.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow(res_rank)
        w.writerow(res_raw)
    print(f"  → Saved: {out}")

if __name__ == "__main__":
    main()
