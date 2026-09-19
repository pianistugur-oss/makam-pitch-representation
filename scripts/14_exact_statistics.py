"""Exact test statistics reported in the paper: Wilcoxon, Mann-Whitney, Kruskal-Wallis (W, U, H, df, exact p, Cohen's d)."""

import csv, sqlite3
from pathlib import Path
from collections import defaultdict

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
KOMA_LARGE, KOMA_SMALL = 27, 9
TET_LARGE,  TET_SMALL  = 6,  2
FACTORS = ["rdir", "diff", "rret", "prox", "clos"]

# ── shared helpers ─────────────────────────────────────────────────────────────

def _int(s):
    try: return int(s)
    except: return None

def read_dat(p):
    rows = []
    with open(p) as f:
        hdr = f.readline().split()
        for line in f:
            pts = line.split()
            row = dict(zip(hdr, pts))
            if row.get("cpitch.ic", "NA") == "NA": continue
            rows.append({
                "mid":  int(row["melody.id"]),
                "nid":  int(row["note.id"]),
                "cp":   int(row["cpitch"]),
                "ic":   float(row["cpitch.ic"]),
                "name": row.get("melody.name", "").strip('"'),
            })
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

def build_piece_delta_ic(makam):
    """Returns np.array of piece-level ΔIC values."""
    dat_k  = sorted((OUTDIR/makam/"koma").glob("*.dat"))[0]
    dat_t  = sorted((OUTDIR/makam/"tet12").glob("*.dat"))[0]
    karar  = get_karar(makam)
    rk, rt = read_dat(dat_k), read_dat(dat_t)
    idx_t  = {(r["mid"], r["nid"]): r["ic"] for r in rt}
    per_k  = defaultdict(list)
    per_t  = defaultdict(list)
    for r in rk:
        ic_t = idx_t.get((r["mid"], r["nid"]))
        if ic_t is None: continue
        if karar.get(r["mid"] - 1) is None: continue
        per_k[r["mid"]].append(r["ic"])
        per_t[r["mid"]].append(ic_t)
    return np.array([np.mean(per_k[m]) - np.mean(per_t[m]) for m in per_k])

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

def pooled_sd(a, b):
    na, nb = len(a), len(b)
    return np.sqrt(
        ((na-1)*np.var(a,ddof=1) + (nb-1)*np.var(b,ddof=1)) / (na+nb-2))

def cohens_d(a, b):
    sp = pooled_sd(a, b)
    return (np.mean(a) - np.mean(b)) / sp if sp > 0 else 0.0

def fmt_p(p):
    return "< .001" if p < .001 else f"= {p:.3f}"

# ── main ───────────────────────────────────────────────────────────────────────

print("Loading piece-level ΔIC …")
dic = {m: build_piece_delta_ic(m) for m in MAKAMS}
for m in MAKAMS:
    print(f"  {LABELS[m]:12s}  n={len(dic[m])}  mean={np.mean(dic[m]):+.5f}")

print("\nLoading piece-level ΔIR (factor decomposition, needed for the ΔIR test) …")
dir_data = {}
for m in MAKAMS:
    delta_ir, _ = compute_ir_with_factors(m)
    dir_data[m] = delta_ir
    print(f"  {LABELS[m]:12s}  n={len(delta_ir)}  mean ΔIR={np.mean(delta_ir):+.6f}")

print("\n" + "="*70)
print("EXACT TEST STATISTICS")
print("="*70)
csv_rows = []

# ① Wilcoxon ΔIC
print(f"\n① Wilcoxon signed-rank  (H₀: median ΔIC = 0)")
print(f"  Note: W = min(T⁺, T⁻) per scipy convention")
print(f"\n  {'Maqam':12s}  {'n':>4}  {'W':>8}  p")
print("  " + "─"*38)
for m in MAKAMS:
    W, p = ss.wilcoxon(dic[m])
    print(f"  {LABELS[m]:12s}  {len(dic[m]):4d}  {W:8.1f}  p {fmt_p(p)}")
    csv_rows.append(dict(test="Wilcoxon", metric="ΔIC", group=LABELS[m],
                         n1=len(dic[m]), n2="", stat=round(W,2), df="", p=round(p,6), d=""))

# ② Wilcoxon ΔIR
print(f"\n② Wilcoxon signed-rank  (H₀: median ΔIR = 0)")
print(f"\n  {'Maqam':12s}  {'n':>4}  {'W':>8}  p")
print("  " + "─"*38)
for m in MAKAMS:
    W, p = ss.wilcoxon(dir_data[m])
    print(f"  {LABELS[m]:12s}  {len(dir_data[m]):4d}  {W:8.1f}  p {fmt_p(p)}")
    csv_rows.append(dict(test="Wilcoxon", metric="ΔIR", group=LABELS[m],
                         n1=len(dir_data[m]), n2="", stat=round(W,2), df="", p=round(p,6), d=""))

# ③ Mann-Whitney
micro_dic = np.concatenate([dic["ussak"], dic["huseyni"]])
micro_dir = np.concatenate([dir_data["ussak"], dir_data["huseyni"]])
ctrl_dic  = dic["nihavent"]
ctrl_dir  = dir_data["nihavent"]
n1, n2    = len(micro_dic), len(ctrl_dic)

print(f"\n③ Mann–Whitney  (microtonal n = {n1}  vs  Nihâvend n = {n2})")
cohens_d_hdr = "Cohen's d"
print(f"\n  {'Metric':6s}  {'U':>9}  p            {cohens_d_hdr:>10}")
print("  " + "─"*48)
for metric, micro, ctrl in [("ΔIC", micro_dic, ctrl_dic), ("ΔIR", micro_dir, ctrl_dir)]:
    U, p = ss.mannwhitneyu(micro, ctrl, alternative="two-sided")
    d    = cohens_d(micro, ctrl)
    print(f"  {metric:6s}  {U:9.1f}  p {fmt_p(p)}  {d:+10.4f}")
    csv_rows.append(dict(test="Mann-Whitney", metric=metric,
                         group=f"micro vs Nihâvend",
                         n1=n1, n2=n2, stat=round(U,2), df="", p=round(p,6), d=round(d,4)))

# ④ Kruskal-Wallis
print(f"\n④ Kruskal–Wallis  (3 maqams, df = 2)")
print(f"\n  {'Metric':6s}  {'H':>9}  p")
print("  " + "─"*30)
for metric, groups in [("ΔIC", [dic[m] for m in MAKAMS]),
                        ("ΔIR", [dir_data[m] for m in MAKAMS])]:
    H, p = ss.kruskal(*groups)
    print(f"  {metric:6s}  {H:9.4f}  p {fmt_p(p)}")
    csv_rows.append(dict(test="Kruskal-Wallis", metric=metric, group="3 maqams",
                         n1="", n2="", stat=round(H,4), df=2, p=round(p,6), d=""))

# ⑤ Cohen's d formula
print(f"""
⑤ Cohen's d formula used throughout:
   d = (mean_A − mean_B) / pooled_SD
   pooled_SD = sqrt( [(n_A−1)·Var_A + (n_B−1)·Var_B] / (n_A+n_B−2) )
   Input values: piece-level means (one value per piece).
   Piece mean = mean of note-level ΔIC (or ΔIR) within that piece.
   This is the standard unbiased pooled-SD estimator (not Hedges' g).""")

# ⑥ Koma53 note counts
print(f"\n⑥ Note counts by absolute Koma53 pitch")
print(f"\n  {'Maqam':12s}  {'Koma53':>7}  {'Pitch label':20s}  {'n notes':>8}")
print("  " + "─"*56)
pitch_labels = {312: "kürdi (Si♭₂)", 313: "segah/nötr-2nd (Si♭₁)"}
for m in MAKAMS:
    dat_k = sorted((OUTDIR/m/"koma").glob("*.dat"))[0]
    recs  = read_dat(dat_k)
    targets = [313] if m == "nihavent" else [312, 313]
    for koma53 in targets:
        count = sum(1 for r in recs if r["cp"] == koma53)
        plabel = pitch_labels.get(koma53, str(koma53))
        print(f"  {LABELS[m]:12s}  {koma53:7d}  {plabel:20s}  {count:8d}")
        csv_rows.append(dict(test="note_count", metric=f"Koma53={koma53}",
                             group=LABELS[m], n1=count, n2="", stat="", df="", p="", d=""))

out = RESULTS / "exact_statistics.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["test","metric","group","n1","n2","stat","df","p","d"])
    w.writeheader(); w.writerows(csv_rows)
print(f"\n  → Saved: {out}")
