"""Table 1: piece-level ΔIC summary (n, mean, median, bootstrap 95% CI) per makam, from existing IDyOM .dat output."""

import csv, sqlite3
from pathlib import Path
from collections import defaultdict

import numpy as np

BASE       = Path("/Users/ugurozalp/makam_beklenti")          # local IDyOM output location; edit for your setup
OUTDIR     = BASE / "data" / "idyom_output"
RESULTS    = Path(__file__).resolve().parent.parent / "results"

MAKAMS  = ["ussak", "huseyni", "nihavent"]
LABELS  = {"ussak": "Uşşak", "huseyni": "Hüseyni", "nihavent": "Nihâvend"}
DATASET = {"ussak": {"koma": 200, "tet12": 201},
           "huseyni": {"koma": 202, "tet12": 203},
           "nihavent": {"koma": 204, "tet12": 205}}

BOOT_N, BOOT_SEED = 5000, 42

def read_dat(p):
    rows = []
    with open(p) as f:
        hdr = f.readline().split()
        for line in f:
            pts = line.split(); row = dict(zip(hdr, pts))
            if row.get("cpitch.ic", "NA") == "NA": continue
            rows.append({"mid": int(row["melody.id"]), "nid": int(row["note.id"]),
                         "ic": float(row["cpitch.ic"])})
    return rows

def piece_dic(makam):
    dat_k = sorted((OUTDIR/makam/"koma").glob("*.dat"))[0]
    dat_t = sorted((OUTDIR/makam/"tet12").glob("*.dat"))[0]
    rk, rt = read_dat(dat_k), read_dat(dat_t)
    idx_t = {(r["mid"], r["nid"]): r["ic"] for r in rt}
    per = defaultdict(list)
    for r in rk:
        ic_t = idx_t.get((r["mid"], r["nid"]))
        if ic_t is None: continue
        per[r["mid"]].append(r["ic"] - ic_t)
    return {mid: np.mean(v) for mid, v in per.items()}

def bootstrap_ci(data, n=BOOT_N, alpha=0.05, seed=BOOT_SEED):
    rng  = np.random.default_rng(seed)
    data = np.asarray(data)
    boot = [np.mean(rng.choice(data, len(data), replace=True)) for _ in range(n)]
    return float(np.percentile(boot, 100*alpha/2)), float(np.percentile(boot, 100*(1-alpha/2)))

print("="*72)
print("TABLE 1 — Piece-level ΔIC summary (n, mean, median, bootstrap 95% CI)")
print(f"(bootstrap: n={BOOT_N} resamples, seed={BOOT_SEED})")
print("="*72)

rows = []
for m in MAKAMS:
    pd_ = list(piece_dic(m).values())
    n      = len(pd_)
    mean_v = float(np.mean(pd_))
    med_v  = float(np.median(pd_))
    lo, hi = bootstrap_ci(pd_)
    print(f"  {LABELS[m]:12s}  n={n:4d}  mean={mean_v:+.4f}  median={med_v:+.4f}  "
          f"95% CI=[{lo:+.4f}, {hi:+.4f}]")
    rows.append({"maqam": LABELS[m], "n": n, "mean_delta_ic": round(mean_v, 6),
                 "median_delta_ic": round(med_v, 6),
                 "ci_lo_95": round(lo, 6), "ci_hi_95": round(hi, 6),
                 "bootstrap_n": BOOT_N, "bootstrap_seed": BOOT_SEED})

out = RESULTS / "piece_level_summary.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["maqam","n","mean_delta_ic","median_delta_ic",
                                      "ci_lo_95","ci_hi_95","bootstrap_n","bootstrap_seed"])
    w.writeheader(); w.writerows(rows)
print(f"\n  → Saved: {out}")
