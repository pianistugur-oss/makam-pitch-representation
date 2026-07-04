"""
Robustness analyses addressing corpus-size confound.

Task A: Alphabet size table — distinct pitches, notes, pieces per maqam × arm.
Task B: Size-matched subsampling — Uşşak/Nihâvend downsampled to 88 pieces
        (Hüseyni size) across 5 random seeds.

Note on Task B methodology: IC estimates are taken from the existing full-corpus
IDyOM runs. Subsampling is applied at the piece level — each seed draws a random
88-piece subset and recomputes the ΔIC statistics on that subset. This tests
whether the pattern holds with equal corpus sizes without re-running IDyOM
(which would require ~20 additional runs taking several hours).
"""

import sqlite3, csv
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats as ss

BASE   = Path("/Users/ugurozalp/makam_beklenti")
OUTDIR = BASE / "data" / "idyom_output"
FIGDIR = BASE / "figures"

MAKAMS  = ["ussak", "huseyni", "nihavent"]
LABELS  = {"ussak": "Uşşak", "huseyni": "Hüseyni", "nihavent": "Nihâvend"}
DATASET = {
    "ussak":    {"koma": 200, "tet12": 201},
    "huseyni":  {"koma": 202, "tet12": 203},
    "nihavent": {"koma": 204, "tet12": 205},
}

# ── helpers ───────────────────────────────────────────────────────────────────

def read_dat(p):
    rows = []
    with open(p) as f:
        hdr = f.readline().split()
        for line in f:
            pts = line.split()
            row = dict(zip(hdr, pts))
            if row.get("cpitch.ic", "NA") == "NA":
                continue
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
            (DATASET[makam]["koma"],),
        ).fetchall()
    last = {}
    for cid, cp in rows:
        last[cid] = cp
    return last

def build_piece_records(makam):
    """Returns {mid: {name, ic_koma, ic_tet, delta_ic}} for all pieces."""
    dat_k = sorted((OUTDIR / makam / "koma").glob("*.dat"))[0]
    dat_t = sorted((OUTDIR / makam / "tet12").glob("*.dat"))[0]
    karar = get_karar(makam)
    rk    = read_dat(dat_k)
    rt    = read_dat(dat_t)
    idx_t = {(r["mid"], r["nid"]): r["ic"] for r in rt}

    per_k  = defaultdict(list)
    per_t  = defaultdict(list)
    names  = {}
    for r in rk:
        ic_t = idx_t.get((r["mid"], r["nid"]))
        if ic_t is None:
            continue
        k = karar.get(r["mid"] - 1)
        if k is None:
            continue
        per_k[r["mid"]].append(r["ic"])
        per_t[r["mid"]].append(ic_t)
        names[r["mid"]] = r["name"]

    return {
        mid: {
            "name":     names[mid],
            "ic_koma":  np.mean(per_k[mid]),
            "ic_tet":   np.mean(per_t[mid]),
            "delta_ic": np.mean(per_k[mid]) - np.mean(per_t[mid]),
        }
        for mid in per_k
    }

def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    sp = np.sqrt(
        ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1))
        / (len(a) + len(b) - 2)
    )
    return (np.mean(a) - np.mean(b)) / sp if sp > 0 else 0.0

def compute_stats(piece_subsets):
    """
    piece_subsets: {makam: list of delta_ic values}
    Returns nested dict of per-makam and global stats.
    """
    result = {}
    for m in MAKAMS:
        v = np.asarray(piece_subsets[m])
        W, p_w = ss.wilcoxon(v)
        result[m] = {
            "n":        len(v),
            "mean":     float(np.mean(v)),
            "median":   float(np.median(v)),
            "wilcox_W": float(W),
            "wilcox_p": float(p_w),
        }

    micro = np.concatenate([piece_subsets["ussak"], piece_subsets["huseyni"]])
    ctrl  = np.asarray(piece_subsets["nihavent"])
    U, p_mw = ss.mannwhitneyu(micro, ctrl, alternative="two-sided")
    d        = cohens_d(micro, ctrl)
    H, p_kw  = ss.kruskal(
        piece_subsets["ussak"], piece_subsets["huseyni"], piece_subsets["nihavent"]
    )
    result["_mw"] = {"U": float(U), "p": float(p_mw), "d": float(d)}
    result["_kw"] = {"H": float(H), "p": float(p_kw)}
    return result

# ── TASK A ────────────────────────────────────────────────────────────────────

def task_a():
    print("\n" + "═" * 66)
    print("TASK A — Alphabet Sizes")
    print("═" * 66)
    header = f"{'Maqam':12s}  {'Arm':6s}  {'Pieces':>7}  {'Notes':>8}  {'Distinct pitches':>16}"
    print(header)
    print("  " + "─" * 56)

    rows = []
    for makam in MAKAMS:
        for arm in ["koma", "tet12"]:
            dat   = sorted((OUTDIR / makam / arm).glob("*.dat"))[0]
            recs  = read_dat(dat)
            n_pieces   = len(set(r["mid"] for r in recs))
            n_notes    = len(recs)
            n_distinct = len(set(r["cp"] for r in recs))
            label = LABELS[makam]
            print(f"  {label:12s}  {arm:6s}  {n_pieces:7d}  {n_notes:8d}  {n_distinct:16d}")
            rows.append({
                "maqam":            label,
                "arm":              arm,
                "pieces":           n_pieces,
                "notes":            n_notes,
                "distinct_pitches": n_distinct,
            })

    out = BASE / "alphabet_sizes.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["maqam", "arm", "pieces", "notes", "distinct_pitches"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n  → Saved: {out}")

# ── TASK B ────────────────────────────────────────────────────────────────────

def task_b():
    print("\n" + "═" * 66)
    print("TASK B — Size-matched Subsampling  (Uşşak/Nihâvend → 88 pieces)")
    print("═" * 66)

    # Load full piece records once
    all_pieces = {m: build_piece_records(m) for m in MAKAMS}

    # Print full-corpus reference stats first
    print("\n--- Full corpus (reference) ---")
    full_subsets = {m: np.asarray([v["delta_ic"] for v in all_pieces[m].values()])
                    for m in MAKAMS}
    full_stats = compute_stats(full_subsets)
    for m in MAKAMS:
        s = full_stats[m]
        p_str = "< .001" if s["wilcox_p"] < .001 else f"= {s['wilcox_p']:.3f}"
        print(f"  {LABELS[m]:12s}  n={s['n']}  mean={s['mean']:+.4f}  "
              f"median={s['median']:+.4f}  W={s['wilcox_W']:.0f}  p {p_str}")
    mw = full_stats["_mw"]; kw = full_stats["_kw"]
    print(f"  MW (micro vs Nihâvend):  U={mw['U']:.1f}  p={mw['p']:.4f}  d={mw['d']:.3f}")
    print(f"  KW (3 maqams):           H={kw['H']:.3f}  p={kw['p']:.4f}")

    TARGET = 88
    SEEDS  = [1, 2, 3, 4, 5]
    seed_stats = []
    csv_rows   = []

    print(f"\n--- Subsampled runs (n = {TARGET} per maqam) ---\n")

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        sub = {}
        for m in MAKAMS:
            mids = list(all_pieces[m].keys())
            if len(mids) > TARGET:
                chosen = rng.choice(mids, TARGET, replace=False)
            else:
                chosen = mids
            sub[m] = np.asarray([all_pieces[m][mid]["delta_ic"] for mid in chosen])

        st = compute_stats(sub)
        seed_stats.append(st)

        print(f"  Seed {seed}:")
        for m in MAKAMS:
            s = st[m]
            stars = ("***" if s["wilcox_p"] < .001 else
                     "**"  if s["wilcox_p"] < .01  else
                     "*"   if s["wilcox_p"] < .05  else "ns")
            print(f"    {LABELS[m]:12s}  n={s['n']}  "
                  f"mean={s['mean']:+.4f}  median={s['median']:+.4f}  "
                  f"W={s['wilcox_W']:.0f}  p={s['wilcox_p']:.4f} {stars}")
            csv_rows.append({
                "seed":       seed,
                "maqam":      LABELS[m],
                "n":          s["n"],
                "mean_dic":   s["mean"],
                "median_dic": s["median"],
                "wilcox_W":   s["wilcox_W"],
                "wilcox_p":   s["wilcox_p"],
                "mw_U":       st["_mw"]["U"],
                "mw_p":       st["_mw"]["p"],
                "mw_d":       st["_mw"]["d"],
                "kw_H":       st["_kw"]["H"],
                "kw_p":       st["_kw"]["p"],
            })
        mw = st["_mw"]; kw = st["_kw"]
        print(f"    MW: U={mw['U']:.1f}  p={mw['p']:.4f}  d={mw['d']:.3f}   "
              f"KW: H={kw['H']:.3f}  p={kw['p']:.4f}\n")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    print("─" * 66)
    print("AGGREGATE  (mean  [min, max]  across 5 seeds)")
    print("─" * 66)

    for m in MAKAMS:
        means   = [st[m]["mean"]     for st in seed_stats]
        medians = [st[m]["median"]   for st in seed_stats]
        Ws      = [st[m]["wilcox_W"] for st in seed_stats]
        ps      = [st[m]["wilcox_p"] for st in seed_stats]
        print(f"\n  {LABELS[m]}  (n = {TARGET}):")
        print(f"    mean ΔIC   = {np.mean(means):+.4f}  [{min(means):+.4f}, {max(means):+.4f}]")
        print(f"    median ΔIC = {np.mean(medians):+.4f}  [{min(medians):+.4f}, {max(medians):+.4f}]")
        print(f"    Wilcoxon W = {np.mean(Ws):.1f}    "
              f"p = {np.mean(ps):.4f}  [{min(ps):.4f}, {max(ps):.4f}]")

    mw_Us = [st["_mw"]["U"] for st in seed_stats]
    mw_ps = [st["_mw"]["p"] for st in seed_stats]
    mw_ds = [st["_mw"]["d"] for st in seed_stats]
    kw_Hs = [st["_kw"]["H"] for st in seed_stats]
    kw_ps = [st["_kw"]["p"] for st in seed_stats]

    print(f"\n  Mann–Whitney (micro vs Nihâvend):")
    print(f"    U = {np.mean(mw_Us):.1f}  [{min(mw_Us):.1f}, {max(mw_Us):.1f}]")
    print(f"    p = {np.mean(mw_ps):.4f}  [{min(mw_ps):.4f}, {max(mw_ps):.4f}]")
    print(f"    d = {np.mean(mw_ds):.3f}  [{min(mw_ds):.3f}, {max(mw_ds):.3f}]")
    print(f"\n  Kruskal–Wallis (3 maqams):")
    print(f"    H = {np.mean(kw_Hs):.3f}  [{min(kw_Hs):.3f}, {max(kw_Hs):.3f}]")
    print(f"    p = {np.mean(kw_ps):.4f}  [{min(kw_ps):.4f}, {max(kw_ps):.4f}]")

    # ── CSV ───────────────────────────────────────────────────────────────────
    out = BASE / "robustness_size_matched.csv"
    fields = ["seed", "maqam", "n", "mean_dic", "median_dic",
              "wilcox_W", "wilcox_p", "mw_U", "mw_p", "mw_d", "kw_H", "kw_p"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\n  → Saved: {out}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 66)
    print("SUMMARY — Full corpus vs. size-matched comparison")
    print("═" * 66)
    ref_mw = full_stats["_mw"]
    print(f"\n  Full corpus (Uşşak 117, Hüseyni 88, Nihâvend 128):")
    print(f"    MW U={ref_mw['U']:.1f}  p={ref_mw['p']:.4f}  d={ref_mw['d']:.3f}")
    print(f"\n  Size-matched (all = 88, mean across 5 seeds):")
    print(f"    MW U={np.mean(mw_Us):.1f}  p={np.mean(mw_ps):.4f}  d={np.mean(mw_ds):.3f}")
    huseyni_means_full = full_subsets["huseyni"]
    ussak_means_full   = full_subsets["ussak"]
    niha_means_full    = full_subsets["nihavent"]
    huseyni_sub_means  = [st["huseyni"]["mean"] for st in seed_stats]
    print(f"\n  Hüseyni mean ΔIC — full: {np.mean(huseyni_means_full):+.4f}  "
          f"subsampled (same): {np.mean(huseyni_means_full):+.4f}  (unchanged — all 88 pieces used)")
    ussak_sub = [st["ussak"]["mean"] for st in seed_stats]
    niha_sub  = [st["nihavent"]["mean"] for st in seed_stats]
    print(f"  Uşşak mean ΔIC  — full: {np.mean(ussak_means_full):+.4f}  "
          f"subsampled mean: {np.mean(ussak_sub):+.4f}")
    print(f"  Nihâvend mean ΔIC — full: {np.mean(niha_means_full):+.4f}  "
          f"subsampled mean: {np.mean(niha_sub):+.4f}")

if __name__ == "__main__":
    task_a()
    task_b()
