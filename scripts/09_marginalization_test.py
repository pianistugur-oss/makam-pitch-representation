"""Marginalizes the comma model's per-event probability distribution to 12-TET classes and compares it event-for-event with the natively-trained 12-TET model."""

import csv, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats as ss

warnings.filterwarnings("ignore")

BASE    = Path("/Users/ugurozalp/makam_beklenti")   # local IDyOM output location; edit for your setup
OUTDIR  = BASE / "data" / "idyom_output"
RESULTS = Path(__file__).resolve().parent.parent / "results"
MAKAMS  = ["ussak", "huseyni", "nihavent"]
LABELS  = {"ussak": "Uşşak", "huseyni": "Hüseyni", "nihavent": "Nihâvend"}

N_BOOT  = 5000
SEED    = 42

# ── helpers ───────────────────────────────────────────────────────────────────

def koma_to_tet(k):
    return round(int(k) * 12 / 53)

def read_dat_with_dist(path):
    """
    Parse a .dat file.
    Returns (records, pitch_cols) where
    records = [{mid, nid, cp, ic, prob_dict}, ...]
    prob_dict = {pitch_value: probability}
    pitch_cols = sorted list of pitch-valued column names found
    """
    path = Path(path)
    with open(path) as f:
        hdr = f.readline().split()

    # identify per-pitch probability columns: 'cpitch.NNN' where NNN is an int
    pitch_cols = []
    for col in hdr:
        parts = col.split(".")
        if len(parts) == 2 and parts[0] == "cpitch":
            try:
                int(parts[1])
                pitch_cols.append(col)
            except ValueError:
                pass

    cp_idx  = hdr.index("cpitch")
    ic_idx  = hdr.index("cpitch.ic")
    mid_idx = hdr.index("melody.id")
    nid_idx = hdr.index("note.id")
    pc_idx  = {col: hdr.index(col) for col in pitch_cols}

    records = []
    with open(path) as f:
        f.readline()                        # skip header
        for line in f:
            pts = line.split()
            if pts[ic_idx] == "NA": continue
            ic_val = float(pts[ic_idx])
            if ic_val != ic_val: continue   # nan guard

            prob = {}
            for col, idx in pc_idx.items():
                v = pts[idx]
                if v != "NA":
                    fv = float(v)
                    if fv == fv:            # nan guard
                        prob[int(col.split(".")[1])] = fv

            records.append({
                "mid":  int(pts[mid_idx]),
                "nid":  int(pts[nid_idx]),
                "cp":   int(pts[cp_idx]),
                "ic":   ic_val,
                "prob": prob,
            })
    return records, pitch_cols


def get_dat_path(makam, arm):
    d = OUTDIR / makam / arm
    return sorted(d.glob("*.dat"))[0]


def marginalize(records_k):
    """
    Compute IC_marg for each comma record.
    Returns records_k with added 'ic_marg' key (None if distribution sums to 0).
    """
    out = []
    for r in records_k:
        prob = r["prob"]
        if not prob:
            out.append({**r, "ic_marg": None}); continue

        # aggregate by 12-TET class
        marg = defaultdict(float)
        for kp, p in prob.items():
            marg[koma_to_tet(kp)] += p

        # observed 12-TET class
        obs_tet = koma_to_tet(r["cp"])
        p_obs = marg.get(obs_tet, 0.0)

        if p_obs <= 0:
            ic_marg = None
        else:
            ic_marg = -np.log2(p_obs)

        out.append({**r, "ic_marg": ic_marg, "obs_tet": obs_tet,
                    "p_marg_sum": sum(marg.values())})
    return out


def bootstrap_ci(diffs, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    means = [np.mean(rng.choice(diffs, len(diffs), replace=True))
             for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return lo, hi


def fmt_p(p):
    return "< .001" if p < .001 else f"= {p:.4f}"


# ── main analysis ─────────────────────────────────────────────────────────────

csv_rows = []

print("Parsing .dat files …")
for m in MAKAMS:
    print(f"  loading {LABELS[m]} koma …", end=" ", flush=True)
    recs_k, kcols = read_dat_with_dist(get_dat_path(m, "koma"))
    print(f"{len(recs_k)} notes, {len(kcols)} pitch cols", end="  |  ")
    print(f"tet12 …", end=" ", flush=True)
    recs_t, tcols = read_dat_with_dist(get_dat_path(m, "tet12"))
    print(f"{len(recs_t)} notes, {len(tcols)} pitch cols")

    # build fast lookup: (mid, nid) → ic_12tet
    tet_lookup = {(r["mid"], r["nid"]): r["ic"] for r in recs_t}

    # marginalize
    recs_m = marginalize(recs_k)

    # collect note-level pairs where both IC_marg and IC_12TET are valid
    note_pairs = []  # (mid, ic_marg, ic_12tet, ic_koma)
    skipped = 0
    for r in recs_m:
        if r["ic_marg"] is None: skipped += 1; continue
        ic12 = tet_lookup.get((r["mid"], r["nid"]))
        if ic12 is None: skipped += 1; continue
        note_pairs.append((r["mid"], r["ic_marg"], ic12, r["ic"]))

    print(f"    matched notes: {len(note_pairs)}  (skipped {skipped})")

    # piece-level aggregation
    piece_marg = defaultdict(list)
    piece_tet  = defaultdict(list)
    for mid, im, it, ik in note_pairs:
        piece_marg[mid].append(im)
        piece_tet[mid].append(it)

    pieces = sorted(piece_marg)
    pm_arr = np.array([np.mean(piece_marg[p]) for p in pieces])
    pt_arr = np.array([np.mean(piece_tet[p])  for p in pieces])
    diff   = pt_arr - pm_arr   # IC_12TET − IC_marg;  positive → marg better

    # statistics
    W, p_w = ss.wilcoxon(diff, alternative="two-sided")
    ci_lo, ci_hi = bootstrap_ci(diff)

    n = len(pieces)
    print(f"\n  ── {LABELS[m]} ──")
    print(f"  n pieces  : {n}")
    print(f"  mean IC_marg  : {np.mean(pm_arr):.5f}")
    print(f"  mean IC_12TET : {np.mean(pt_arr):.5f}")
    diff_mean = np.mean(diff); diff_med = np.median(diff)
    print(f"  mean diff (12TET − marg): {diff_mean:+.5f}")
    print(f"  median diff             : {diff_med:+.5f}")
    print(f"  Wilcoxon W={W:.1f}, p {fmt_p(p_w)}")
    print(f"  Bootstrap 95% CI: [{ci_lo:+.5f}, {ci_hi:+.5f}]")
    print(f"  → {'Comma marg BETTER than 12TET' if diff_mean > 0 else 'Comma marg WORSE (or equal)'}")

    # store for KW
    csv_rows.append(("PIECE_DIFF", LABELS[m], diff.tolist()))

    # CSV rows
    for key, val in [
        ("n_pieces",      n),
        ("n_notes",       len(note_pairs)),
        ("mean_ic_marg",  round(float(np.mean(pm_arr)), 6)),
        ("mean_ic_tet12", round(float(np.mean(pt_arr)), 6)),
        ("mean_diff",     round(float(diff_mean), 6)),
        ("median_diff",   round(float(diff_med), 6)),
        ("wilcoxon_W",    round(float(W), 4)),
        ("wilcoxon_p",    round(float(p_w), 6)),
        ("ci_lo_95",      round(float(ci_lo), 6)),
        ("ci_hi_95",      round(float(ci_hi), 6)),
    ]:
        csv_rows.append(("SUMMARY", LABELS[m], key, str(val)))

# ── Kruskal–Wallis across maqams ─────────────────────────────────────────────
print("\n" + "═"*60)
print("Kruskal–Wallis  (piece-level diff across 3 maqams)")
diffs_by_makam = {row[1]: np.array(row[2]) for row in csv_rows if row[0] == "PIECE_DIFF"}
arrays = [diffs_by_makam[LABELS[m]] for m in MAKAMS]
H_kw, p_kw = ss.kruskal(*arrays)
print(f"  H = {H_kw:.4f},  p {fmt_p(p_kw)}")

# ── note-level: do prob distributions sum to 1? (sanity) ─────────────────────
print("\n" + "═"*60)
print("Sanity check — comma distribution sum per note")
for m in MAKAMS:
    recs_k, _ = read_dat_with_dist(get_dat_path(m, "koma"))
    recs_m    = marginalize(recs_k)
    sums = [r.get("p_marg_sum", None) for r in recs_m if r.get("p_marg_sum") is not None]
    arr  = np.array(sums)
    print(f"  {LABELS[m]:12s}  mean_sum={np.mean(arr):.4f}  min={np.min(arr):.4f}  max={np.max(arr):.4f}")

# ── Save CSV ──────────────────────────────────────────────────────────────────
out = RESULTS / "marginalization_test.csv"
fields = ["section", "maqam", "key", "value"]
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for row in csv_rows:
        if row[0] == "SUMMARY":
            w.writerow({"section": row[0], "maqam": row[1], "key": row[2], "value": row[3]})
        # PIECE_DIFF arrays: write one row per piece
        elif row[0] == "PIECE_DIFF":
            for i, v in enumerate(row[2]):
                w.writerow({"section": row[0], "maqam": row[1],
                            "key": f"piece_{i}", "value": round(float(v), 6)})
    w.writerow({"section": "KW", "maqam": "all", "key": "KW_H",  "value": round(float(H_kw), 6)})
    w.writerow({"section": "KW", "maqam": "all", "key": "KW_p",  "value": round(float(p_kw), 6)})

print(f"\n→ Saved: {out}")
