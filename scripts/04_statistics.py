# Compute piece-level and degree-level ΔIC statistics (Wilcoxon, Mann-Whitney, Kruskal-Wallis, Cohen's d).

import sqlite3
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats as ss

# ── Configure this path before running ────────────────────────────────────────
OUTPUT_DIR = Path("data/idyom_output")

DATASET_IDS = {
    "ussak":    {"koma": 200, "tet12": 201},
    "huseyni":  {"koma": 202, "tet12": 203},
    "nihavent": {"koma": 204, "tet12": 205},
}

# ── Data helpers ──────────────────────────────────────────────────────────────

def read_dat(p):
    rows = []
    with open(p) as f:
        hdr = f.readline().split()
        for line in f:
            pts = line.split(); row = dict(zip(hdr, pts))
            if row.get("cpitch.ic", "NA") == "NA": continue
            rows.append({"mid": int(row["melody.id"]),
                         "nid": int(row["note.id"]),
                         "cp":  int(row["cpitch"]),
                         "ic":  float(row["cpitch.ic"]),
                         "name": row.get("melody.name", "").strip('"')})
    return rows

def get_karar(makam):
    db = OUTPUT_DIR / makam / "koma" / "idyom.db"
    with sqlite3.connect(db) as c:
        rows = c.execute("SELECT COMPOSITION_ID, CPITCH FROM mtp_event "
                         "WHERE DATASET_ID=? ORDER BY COMPOSITION_ID, ONSET",
                         (DATASET_IDS[makam]["koma"],)).fetchall()
    last = {}
    for cid, cp in rows: last[cid] = cp
    return last   # {comp_id_db (0-based): last_koma}

def build_records(makam):
    dat_k = sorted((OUTPUT_DIR / makam / "koma").glob("*.dat"))[0]
    dat_t = sorted((OUTPUT_DIR / makam / "tet12").glob("*.dat"))[0]
    karar = get_karar(makam)
    rk    = read_dat(dat_k); rt = read_dat(dat_t)
    idx_t = {(r["mid"], r["nid"]): r["ic"] for r in rt}
    recs  = []
    for r in rk:
        ic_t = idx_t.get((r["mid"], r["nid"]))
        if ic_t is None: continue
        k = karar.get(r["mid"] - 1)   # DB is 0-based; dat melody.id is 1-based
        if k is None: continue
        recs.append({"mid": r["mid"], "name": r["name"],
                     "cp_koma": r["cp"],
                     "delta_ic": r["ic"] - ic_t,
                     "deg": (r["cp"] - k) % 53})
    return recs

def piece_dic(recs):
    per = defaultdict(list); names = {}
    for r in recs:
        per[r["mid"]].append(r["delta_ic"]); names[r["mid"]] = r["name"]
    return {(m, names[m]): np.mean(v) for m, v in per.items()}

def bootstrap_ci(data, n=5000, seed=42):
    rng  = np.random.default_rng(seed); data = np.asarray(data)
    boot = [np.mean(rng.choice(data, len(data), replace=True)) for _ in range(n)]
    return np.percentile(boot, 2.5), np.percentile(boot, 97.5)

def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    sp = np.sqrt(((len(a)-1)*np.var(a,ddof=1)+(len(b)-1)*np.var(b,ddof=1))/(len(a)+len(b)-2))
    return (np.mean(a)-np.mean(b))/sp if sp > 0 else 0.0

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    all_recs  = {m: build_records(m) for m in ["ussak","huseyni","nihavent"]}
    piece_data = {}

    print("=== Piece-level ΔIC statistics ===\n")
    for makam in ["ussak","huseyni","nihavent"]:
        vals = list(piece_dic(all_recs[makam]).values())
        piece_data[makam] = vals
        lo, hi = bootstrap_ci(vals)
        W, p   = ss.wilcoxon(vals)
        stars  = "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "ns"
        print(f"  {makam:10s}  n={len(vals):3d}  "
              f"mean={np.mean(vals):+.4f}  median={np.median(vals):+.4f}  "
              f"95%CI=[{lo:+.4f},{hi:+.4f}]  Wilcoxon p={p:.4f} {stars}")

    micro = piece_data["ussak"] + piece_data["huseyni"]
    ctrl  = piece_data["nihavent"]
    H, p_kw = ss.kruskal(*[piece_data[m] for m in ["ussak","huseyni","nihavent"]])
    U, p_mw = ss.mannwhitneyu(micro, ctrl, alternative="two-sided")
    d       = cohens_d(micro, ctrl)
    print(f"\n  Kruskal-Wallis (3 maqams): H={H:.3f}  p={p_kw:.4f}")
    print(f"  Mann-Whitney (microtonal vs Nihavend): U={U:.1f}  p={p_mw:.4f}  Cohen's d={d:.3f}")

    print("\n=== Degree-level ΔIC (n ≥ 50, d=7 and d=8) ===\n")
    print(f"  {'maqam':10s}  {'d':>3s}  {'n':>6s}  {'mean ΔIC':>10s}")
    print("  " + "─"*36)
    for makam in ["ussak","huseyni"]:
        dd = defaultdict(list)
        for r in all_recs[makam]: dd[r["deg"]].append(r["delta_ic"])
        for d in sorted(dd):
            if len(dd[d]) < 50: continue
            flag = "  ← neutral 2nd" if d==8 else ("  ← kurdi" if d==7 else "")
            print(f"  {makam:10s}  {d:3d}  {len(dd[d]):6d}  {np.mean(dd[d]):+10.4f}{flag}")

if __name__ == "__main__":
    main()
