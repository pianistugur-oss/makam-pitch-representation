"""Decomposes piece-level ΔIR into the five Schellenberg (1996) implication-realization factors and reports each factor's share of the total."""

import csv, sqlite3
from pathlib import Path
from collections import defaultdict

import numpy as np

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
KOMA_LARGE, KOMA_SMALL = 27, 9
TET_LARGE,  TET_SMALL  = 6,  2
FACTORS = ["rdir", "diff", "rret", "prox", "clos"]
FACTOR_NAMES = {
    "rdir": "Registral Direction",
    "diff": "Intervallic Difference",
    "rret": "Registral Return",
    "prox": "Proximity",
    "clos": "Closure",
}

# ── shared helpers ─────────────────────────────────────────────────────────────

def _int(s):
    try: return int(s)
    except: return None

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

def _parse_txt(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            pts = line.rstrip("\n").split("\t")
            if len(pts) < 9 or _int(pts[1]) != 9: continue
            koma = _int(pts[4])
            if not koma or koma <= 0: continue
            rows.append((koma, round(koma * 12 / 53)))
    return rows

def get_pieces(makam):
    db = OUTDIR / makam / "koma" / "idyom.db"
    with sqlite3.connect(db) as c:
        return c.execute(
            "SELECT COMPOSITION_ID, DESCRIPTION FROM mtp_composition "
            "WHERE DATASET_ID=? ORDER BY COMPOSITION_ID",
            (DATASET[makam]["koma"],)).fetchall()

def _ir_score_factors(I1, I2, large, small):
    """Returns (rdir, diff, rret, prox, clos) as individual floats."""
    a1, a2 = abs(I1), abs(I2)
    same   = (I1 > 0 and I2 > 0) or (I1 < 0 and I2 < 0)
    rdir = (1.0 if same else 0.0) if a1 > large else \
           (0.0 if same else 1.0) if a1 <= small else 0.0
    diff = 1.0 if a2 < a1 else 0.0
    rret = 1.0 if abs(I1 + I2) <= small else 0.0
    prox = max(0.0, 1.0 - a2 / large) if large > 0 else 0.0
    clos = 1.0 if (not same) and (a2 <= small) else 0.0
    return rdir, diff, rret, prox, clos

def compute_ir_with_factors(makam):
    """Returns delta_ir (piece-level array) and {factor: piece-level Δfactor array}."""
    pieces    = get_pieces(makam)
    karar_map = get_karar(makam)
    piece_dir = []
    piece_fac = {f: [] for f in FACTORS}

    for cid_db, name in pieces:
        txt = SYMBTR_DIR / (name + ".txt")
        if not txt.exists(): continue
        notes = _parse_txt(txt)
        if len(notes) < 3: continue
        karar = karar_map.get(cid_db)
        if karar is None: continue
        komas  = [n[0] for n in notes]
        tet12s = [n[1] for n in notes]

        fac_k = {f: [] for f in FACTORS}
        fac_t = {f: [] for f in FACTORS}
        for i in range(len(notes) - 2):
            kf = _ir_score_factors(komas[i+1]-komas[i],   komas[i+2]-komas[i+1],   KOMA_LARGE, KOMA_SMALL)
            tf = _ir_score_factors(tet12s[i+1]-tet12s[i], tet12s[i+2]-tet12s[i+1], TET_LARGE,  TET_SMALL)
            for j, f in enumerate(FACTORS):
                fac_k[f].append(kf[j]); fac_t[f].append(tf[j])

        if not fac_k["rdir"]: continue
        piece_dir.append(sum(np.mean(fac_k[f]) - np.mean(fac_t[f]) for f in FACTORS))
        for f in FACTORS:
            piece_fac[f].append(np.mean(fac_k[f]) - np.mean(fac_t[f]))

    return np.array(piece_dir), {f: np.array(piece_fac[f]) for f in FACTORS}

# ── main ───────────────────────────────────────────────────────────────────────

print("Loading piece-level ΔIR + IR factors …")
ir_store = {}   # {makam: (delta_ir_array, {factor: array})}
for m in MAKAMS:
    delta_ir, fac_deltas = compute_ir_with_factors(m)
    ir_store[m] = (delta_ir, fac_deltas)
    print(f"  {LABELS[m]:12s}  n={len(delta_ir)}  mean ΔIR={np.mean(delta_ir):+.6f}")

print("\n" + "="*70)
print("ΔIR FACTOR DECOMPOSITION")
print("="*70)
csv_rows = []

print(f"\n  {'Maqam':12s}  {'Factor':24s}  {'Δfactor':>10}  {'% of ΔIR':>10}")
print("  " + "─"*64)

for m in MAKAMS:
    delta_ir, fac_deltas = ir_store[m]
    total_dir = float(np.mean(delta_ir))
    n_pieces  = len(delta_ir)
    factor_means = {f: float(np.mean(fac_deltas[f])) for f in FACTORS}
    check_sum    = sum(factor_means.values())

    print(f"\n  {LABELS[m]:12s}  n = {n_pieces}  "
          f"mean ΔIR = {total_dir:+.6f}  (factor sum = {check_sum:+.6f})")
    for f in FACTORS:
        pct = factor_means[f] / total_dir * 100 if total_dir != 0 else float("nan")
        print(f"    {FACTOR_NAMES[f]:24s}  {factor_means[f]:+10.6f}  {pct:9.1f}%")
        csv_rows.append(dict(maqam=LABELS[m], n_pieces=n_pieces, factor=FACTOR_NAMES[f],
                             delta_factor=round(factor_means[f],7),
                             mean_total_dir=round(total_dir,7),
                             pct_of_total_dir=round(pct,2)))
    print(f"    {'TOTAL':24s}  {total_dir:+10.6f}  {'100.0':>9}%")

print(f"\n  ─── Dominant factor per maqam ───")
for m in MAKAMS:
    _, fac_deltas = ir_store[m]
    fac_means = {f: float(np.mean(fac_deltas[f])) for f in FACTORS}
    dom = max(FACTORS, key=lambda f: fac_means[f])
    print(f"  {LABELS[m]:12s}: {FACTOR_NAMES[dom]} "
          f"({fac_means[dom]:+.6f})")

out = RESULTS / "ir_factor_decomposition.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["maqam","n_pieces","factor",
                                      "delta_factor","mean_total_dir","pct_of_total_dir"])
    w.writeheader(); w.writerows(csv_rows)
print(f"\n  → Saved: {out}")
