"""Tests an alternative 12-TET mapping (SymbTr NotaAE simple-Western respelling) against the round(Koma53×12/53) convention and reports where they diverge and how much that changes ΔIC."""

import csv, re, sqlite3
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from scipy import stats as ss

BASE       = Path("/Users/ugurozalp/makam_beklenti")          # local IDyOM output location; edit for your setup
SYMBTR_DIR = Path("/Users/ugurozalp/Downloads/SymbTr-master/txt")  # local SymbTr checkout; edit for your setup
OUTDIR     = BASE / "data" / "idyom_output"
RESULTS    = Path(__file__).resolve().parent.parent / "results"

MAKAMS  = ["ussak", "huseyni", "nihavent"]
LABELS  = {"ussak": "Uşşak", "huseyni": "Hüseyni", "nihavent": "Nihâvend"}
MICRO   = ["ussak", "huseyni"]
DATASET = {"ussak": {"koma": 200, "tet12": 201},
           "huseyni": {"koma": 202, "tet12": 203},
           "nihavent": {"koma": 204, "tet12": 205}}

def _int(s):
    try: return int(s)
    except: return None

def koma_to_tet(k): return round(int(k) * 12 / 53)

def fmt_p(p): return "< .001" if p < .001 else f"= {p:.4f}"

# ── data loading ─────────────────────────────────────────────────────────────

def read_dat_with_dist(path):
    """Parse .dat -> (records, pitch_cols). records carry a 'prob' dict."""
    path = Path(path)
    with open(path) as f: hdr = f.readline().split()
    pitch_cols = []
    for col in hdr:
        parts = col.split(".")
        if len(parts) == 2 and parts[0] == "cpitch":
            try: int(parts[1]); pitch_cols.append(col)
            except ValueError: pass
    cp_idx, ic_idx = hdr.index("cpitch"), hdr.index("cpitch.ic")
    mid_idx, nid_idx = hdr.index("melody.id"), hdr.index("note.id")
    pc_idx = {col: hdr.index(col) for col in pitch_cols}
    records = []
    with open(path) as f:
        f.readline()
        for line in f:
            pts = line.split()
            if pts[ic_idx] == "NA": continue
            ic_val = float(pts[ic_idx])
            if ic_val != ic_val: continue
            prob = {}
            for col, idx in pc_idx.items():
                v = pts[idx]
                if v != "NA":
                    fv = float(v)
                    if fv == fv and fv > 0: prob[int(col.split(".")[1])] = fv
            records.append({"mid": int(pts[mid_idx]), "nid": int(pts[nid_idx]),
                            "cp": int(pts[cp_idx]), "ic": ic_val, "prob": prob})
    return records, pitch_cols

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

def get_dat_path(makam, arm):
    return sorted((OUTDIR / makam / arm).glob("*.dat"))[0]

print("Pre-loading IDyOM output …")
CORPUS = {}
for m in MAKAMS:
    rk, _  = read_dat_with_dist(get_dat_path(m, "koma"))
    rt, _  = read_dat_with_dist(get_dat_path(m, "tet12"))
    tet_lk = {(r["mid"], r["nid"]): r["ic"] for r in rt}
    karar  = get_karar(m)
    merged = []
    for r in rk:
        ic_t = tet_lk.get((r["mid"], r["nid"]))
        if ic_t is None: continue
        k = karar.get(r["mid"] - 1)
        if k is None: continue
        merged.append({**r, "ic_tet": ic_t, "cp_tet": koma_to_tet(r["cp"]),
                       "deg": (r["cp"] - k) % 53})
    CORPUS[m] = {"recs": merged}
    print(f"  {LABELS[m]:12s}  {len(merged)} notes")

# ═════════════════════════════════════════════════════════════════════════════
# Alternative 12-TET mapping: NotaAE simple-Western vs round(Koma53×12/53)
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "="*72)
print("ALTERNATIVE 12-TET MAPPING  (NotaAE simple-Western vs round(K×12/53))")
print("="*72)

WEST_PC = {'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}

def nota_ae_to_midi(nota):
    """
    Parse NotaAE like 'A4', 'B4b1', 'F4#4', 'G4'.
    Returns (round_midi, simple_midi) where:
      round_midi  = round(koma_pitch_implied * 12/53)  -- from comma count
      simple_midi = base_note + (-1 if 'b', +1 if '#', 0 if natural)
    """
    m = re.match(r'^([A-G])(\d+)(b|#)?(\d+)?$', nota.strip())
    if not m: return None, None
    letter, octave_s, acc_type, acc_count_s = m.groups()
    octave = int(octave_s)
    base_midi = 12 * (octave + 1) + WEST_PC[letter]
    acc_count = int(acc_count_s) if acc_count_s else (1 if acc_type else 0)
    comma_shift = -acc_count if acc_type == 'b' else (acc_count if acc_type == '#' else 0)
    base_koma = round(base_midi * 53 / 12)
    koma53    = base_koma + comma_shift
    round_midi  = round(koma53 * 12 / 53)
    simple_midi = base_midi + (-1 if acc_type == 'b' else +1 if acc_type == '#' else 0)
    return round_midi, simple_midi

def parse_symbtr_ae(path):
    """Returns list of (koma53, nota_ae_str) for pitched notes."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            pts = line.rstrip("\n").split("\t")
            if len(pts) < 5 or _int(pts[1]) != 9: continue
            koma = _int(pts[4]); nota_ae = pts[3].strip()
            if koma and koma > 0 and nota_ae: rows.append((koma, nota_ae))
    return rows

print("\n  Scanning SymbTr TXT files …")
koma_ae_info = defaultdict(lambda: {"cnt": Counter(), "nota": Counter(),
                                     "round": None, "simple": None})
csv_divergers = []

for m in MAKAMS:
    txt_files = sorted(SYMBTR_DIR.glob(f"{m}--*.txt"))
    print(f"  {LABELS[m]:12s}  {len(txt_files)} files …", end="  ")
    n_total = 0; n_div = 0
    divs = defaultdict(int)
    for path in txt_files:
        for koma, nota_ae in parse_symbtr_ae(path):
            r_midi, s_midi = nota_ae_to_midi(nota_ae)
            if r_midi is None or s_midi is None: continue
            n_total += 1
            expected_round = koma_to_tet(koma)
            if s_midi != expected_round:
                n_div += 1
                divs[koma] += 1
            koma_ae_info[koma]["cnt"][m] += 1
            koma_ae_info[koma]["nota"].update([nota_ae])
            koma_ae_info[koma]["round"]  = expected_round
            koma_ae_info[koma]["simple"] = s_midi
    print(f"notes={n_total}  diverge={n_div} ({100*n_div/max(n_total,1):.1f}%)")
    for koma, cnt in sorted(divs.items()):
        csv_divergers.append({"maqam": LABELS[m], "koma53": koma,
                              "round_tet": koma_to_tet(koma),
                              "simple_tet": koma_ae_info[koma]["simple"],
                              "nota_ae": list(koma_ae_info[koma]["nota"].keys()),
                              "n_notes_in_makam": cnt})

print(f"\n  Diverging koma53 pitches (round_tet != simple_tet):")
divergers = sorted([(k, info) for k, info in koma_ae_info.items()
                    if info["round"] != info["simple"]], key=lambda x: x[0])
print(f"  {'Koma53':>8}  {'NotaAE':>10}  {'round_tet':>10}  {'simple_tet':>11}  "
      f"notes_Uşşak  notes_Hüseyni  notes_Nihâvend")
print("  " + "─"*80)
for koma, info in divergers:
    nota_str = ",".join(info["nota"].keys())
    nu, nh, nn = (info["cnt"].get(x, 0) for x in ("ussak", "huseyni", "nihavent"))
    print(f"  {koma:8d}  {nota_str:>10}  {info['round']:10d}  {info['simple']:11d}  "
          f"{nu:12d}  {nh:13d}  {nn:14d}")

# Impact on ΔIC: for diverging notes, IC_tet under the alternative (simple)
# mapping = -log2 P_tet(simple), read from the existing 12-TET .dat distribution
# (APPROX: uses the model's fixed prediction, not a retrained model).
print(f"\n  Impact on ΔIC: substitute IC_tet for simple_tet class at diverging notes (APPROX)")

csv_impact = []
for m in MAKAMS:
    recs_t, _ = read_dat_with_dist(get_dat_path(m, "tet12"))
    tet_prob = {(r["mid"], r["nid"]): r["prob"] for r in recs_t}

    recs = CORPUS[m]["recs"]
    div_komas = {k for k, info in koma_ae_info.items()
                if info["round"] != info["simple"] and info["cnt"].get(m, 0) > 0}

    n_div = n_chg = 0
    per_mid_orig, per_mid_alt = defaultdict(list), defaultdict(list)
    for r in recs:
        orig_dic = r["ic"] - r["ic_tet"]
        if r["cp"] in div_komas:
            n_div += 1
            prob_t = tet_prob.get((r["mid"], r["nid"]), {})
            simple = koma_ae_info[r["cp"]]["simple"]
            p_simple = prob_t.get(simple, 0)
            if p_simple > 0:
                alt_dic = r["ic"] - (-np.log2(p_simple))
                n_chg += 1
            else:
                alt_dic = orig_dic
        else:
            alt_dic = orig_dic
        per_mid_orig[r["mid"]].append(orig_dic)
        per_mid_alt[r["mid"]].append(alt_dic)

    pieces  = sorted(per_mid_orig)
    pm_orig = np.array([np.mean(per_mid_orig[p]) for p in pieces])
    pm_alt  = np.array([np.mean(per_mid_alt[p]) for p in pieces])
    diff    = pm_alt - pm_orig

    if np.any(diff != 0):
        W, p_w = ss.wilcoxon(diff, alternative="two-sided")
    else:
        W, p_w = np.nan, np.nan

    print(f"\n  {LABELS[m]}")
    print(f"    diverging notes: {n_div}  (with valid lookup: {n_chg})")
    print(f"    piece-level orig ΔIC: {pm_orig.mean():+.5f}")
    print(f"    piece-level alt  ΔIC: {pm_alt.mean():+.5f}")
    print(f"    diff (alt-orig) mean={diff.mean():+.5f} median={np.median(diff):+.5f}")
    if W == W:
        print(f"    Wilcoxon W={W:.1f}, p {fmt_p(p_w)}")
    else:
        print(f"    Wilcoxon: no variation (all diffs zero)")

    for key, val in [("n_div", n_div), ("n_chg", n_chg),
                     ("orig_mean", pm_orig.mean()), ("alt_mean", pm_alt.mean()),
                     ("diff_mean", diff.mean()), ("diff_med", np.median(diff))]:
        csv_impact.append({"maqam": LABELS[m], "key": key, "value": round(float(val), 6)})
    if W == W:
        csv_impact.append({"maqam": LABELS[m], "key": "wilcoxon_W", "value": round(float(W), 4)})
        csv_impact.append({"maqam": LABELS[m], "key": "wilcoxon_p", "value": round(float(p_w), 6)})

# ── save ──────────────────────────────────────────────────────────────────────

out = RESULTS / "alt_mapping.csv"
all_rows = []
for row in csv_divergers:
    all_rows.append({"section": "divergers", "maqam": row["maqam"],
                     "koma53": str(row["koma53"]),
                     "key": f"round={row['round_tet']}_simple={row['simple_tet']}",
                     "value": str(row["n_notes_in_makam"])})
for row in csv_impact:
    all_rows.append({"section": "impact", "maqam": row["maqam"],
                     "koma53": "all", "key": row["key"], "value": row["value"]})

fields = ["section", "maqam", "koma53", "key", "value"]
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for row in all_rows:
        w.writerow({k: row.get(k, "") for k in fields})
print(f"\n  → Saved: {out}  ({len(all_rows)} rows)")
