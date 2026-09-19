"""Contextual/co-occurrence analysis of Hüseyni d=37 vs. reference d=39 (Eviç), plus a piece-length confound check via logistic regression."""
import csv
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from scipy import stats as ss
import statsmodels.api as sm

SYMBTR_DIR = Path("/Users/ugurozalp/Downloads/SymbTr-master/txt")   # local SymbTr checkout; edit for your setup
RESULTS    = Path(__file__).resolve().parent.parent / "results"
WHOLE_TICKS = 384
WINDOW = 16
CLUSTER_GAP = 3

TARGETS = [
    (37, "2 commas above Acem", [35, 36, 39, 31, 44]),
    (39, "Eviç (reference)",     [35, 36, 37, 31, 44]),
]

def _int(s):
    try: return int(s)
    except: return None

def _float(s):
    try: return float(s)
    except: return None

def parse_piece(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 13: continue
            if _int(parts[1]) != 9: continue
            koma = _int(parts[4])
            if koma is None or koma <= 0: continue
            pay, payda = _int(parts[6]) or 0, _int(parts[7]) or 0
            dur = max(1, round(pay * WHOLE_TICKS / payda)) if payda > 0 else 1
            offset = _float(parts[12])
            rows.append({"koma": koma, "dur": dur, "offset": offset})
    return rows

def load_corpus(makam):
    pieces = {}
    for txt in sorted(SYMBTR_DIR.glob(f"{makam}--*.txt")):
        rows = parse_piece(txt)
        if len(rows) < 3: continue
        karar = rows[-1]["koma"]
        for i, r in enumerate(rows):
            r["idx"] = i
            r["deg"] = (r["koma"] - karar) % 53
        pieces[txt.stem] = rows
    return pieces

def median(xs): return float(np.median(xs)) if xs else float("nan")

print("Loading Hüseyni corpus …")
pieces = load_corpus("huseyni")
total_pieces = len(pieces)
n_notes = sum(len(v) for v in pieces.values())
print(f"  {total_pieces} pieces, {n_notes} notes")

csv_rows = []
def add(deg, section, key, value):
    csv_rows.append({"maqam": "huseyni", "degree": deg, "section": section,
                     "key": key, "value": value})

summary_table = []

for deg, name, companions in TARGETS:
    all_durs = [r["dur"] for p in pieces.values() for r in p]
    corpus_mean_dur = np.mean(all_durs)

    print("\n" + "="*78)
    print(f"HÜSEYNİ  d={deg}  ({name})   companions tested: {companions}")
    print("="*78)

    # 1. Distribution
    per_piece_count = {}
    for pname, rows in pieces.items():
        c = sum(1 for r in rows if r["deg"] == deg)
        if c > 0: per_piece_count[pname] = c
    n_pieces_with = len(per_piece_count)
    counts = list(per_piece_count.values())
    med_c = median(counts)
    max_c = max(counts) if counts else 0
    total_occ = sum(counts)

    print(f"\n1. DISTRIBUTION")
    print(f"   pieces with d={deg}: {n_pieces_with} / {total_pieces} "
          f"({100*n_pieces_with/total_pieces:.1f}%)")
    print(f"   notes per piece (among pieces containing it): median={med_c:.1f}, max={max_c}")
    print(f"   total occurrences: {total_occ}")

    add(deg, "1_distribution", "n_pieces_with", n_pieces_with)
    add(deg, "1_distribution", "total_pieces", total_pieces)
    add(deg, "1_distribution", "pct_pieces_with", round(100*n_pieces_with/total_pieces, 2))
    add(deg, "1_distribution", "median_notes_per_piece", med_c)
    add(deg, "1_distribution", "max_notes_per_piece", max_c)
    add(deg, "1_distribution", "total_occurrences", total_occ)

    # 2. Companion / tetrachord signature test
    print(f"\n2. COMPANION TEST (±{WINDOW}-note window + piece-level presence)")
    comp_results = {}
    for comp in companions:
        base_pieces_with_comp = sum(1 for rows in pieces.values()
                                     if any(r["deg"] == comp for r in rows))
        base_rate_piece = base_pieces_with_comp / total_pieces

        co_piece = sum(1 for pname in per_piece_count
                       if any(r["deg"] == comp for r in pieces[pname]))
        co_piece_rate = co_piece / n_pieces_with if n_pieces_with else float("nan")

        n_occ_with_window_comp = 0
        n_occ_total = 0
        for pname, rows in pieces.items():
            degs_ = [r["deg"] for r in rows]
            n = len(degs_)
            for i, d in enumerate(degs_):
                if d != deg: continue
                n_occ_total += 1
                lo, hi = max(0, i - WINDOW), min(n, i + WINDOW + 1)
                if comp in degs_[lo:i] + degs_[i+1:hi]:
                    n_occ_with_window_comp += 1
        window_rate = n_occ_with_window_comp / n_occ_total if n_occ_total else float("nan")

        n_base_total = 0
        n_base_with_comp = 0
        for pname, rows in pieces.items():
            degs_ = [r["deg"] for r in rows]
            n = len(degs_)
            for i in range(n):
                n_base_total += 1
                lo, hi = max(0, i - WINDOW), min(n, i + WINDOW + 1)
                if comp in degs_[lo:i] + degs_[i+1:hi]:
                    n_base_with_comp += 1
        window_base_rate = n_base_with_comp / n_base_total if n_base_total else float("nan")
        lift = window_rate / window_base_rate if window_base_rate > 0 else float("nan")

        comp_results[comp] = dict(base_rate_piece=base_rate_piece, co_piece_rate=co_piece_rate,
                                   window_rate=window_rate, window_base_rate=window_base_rate,
                                   lift=lift)

        flag = "  <== SUBSTITUTION TEST" if (deg == 37 and comp == 39) else ""
        print(f"   companion d={comp}:{flag}")
        print(f"     corpus base rate (piece-level presence)  = {100*base_rate_piece:.1f}%")
        print(f"     co-occurrence rate (piece-level)         = {100*co_piece_rate:.1f}%  "
              f"({co_piece}/{n_pieces_with} pieces)")
        print(f"     window co-occurrence rate (±{WINDOW})        = {100*window_rate:.1f}%  "
              f"({n_occ_with_window_comp}/{n_occ_total})")
        print(f"     corpus-wide window base rate             = {100*window_base_rate:.1f}%")
        print(f"     lift                                     = {lift:.2f}x")

        add(deg, "2_companion", f"comp{comp}_base_rate_piece_pct", round(100*base_rate_piece,2))
        add(deg, "2_companion", f"comp{comp}_cooccur_piece_pct", round(100*co_piece_rate,2))
        add(deg, "2_companion", f"comp{comp}_window_rate_pct", round(100*window_rate,2))
        add(deg, "2_companion", f"comp{comp}_window_base_rate_pct", round(100*window_base_rate,2))
        add(deg, "2_companion", f"comp{comp}_lift", round(lift,3) if lift==lift else "")

    # 3. Temporal clustering
    run_lengths = []
    for pname in per_piece_count:
        idxs = sorted(r["idx"] for r in pieces[pname] if r["deg"] == deg)
        run = [idxs[0]]
        for prev, cur in zip(idxs, idxs[1:]):
            if cur - prev <= CLUSTER_GAP: run.append(cur)
            else: run_lengths.append(len(run)); run = [cur]
        run_lengths.append(len(run))
    n_runs = len(run_lengths)
    n_isolated = sum(1 for r in run_lengths if r == 1)
    mean_run = np.mean(run_lengths) if run_lengths else float("nan")
    max_run = max(run_lengths) if run_lengths else 0

    print(f"\n3. TEMPORAL CLUSTERING (gap<={CLUSTER_GAP})")
    print(f"   n runs={n_runs}, isolated={n_isolated} ({100*n_isolated/n_runs:.1f}%), "
          f"mean run={mean_run:.2f}, max={max_run}")

    add(deg, "3_clustering", "n_runs", n_runs)
    add(deg, "3_clustering", "pct_isolated", round(100*n_isolated/n_runs,1))
    add(deg, "3_clustering", "mean_run_length", round(float(mean_run),3))
    add(deg, "3_clustering", "max_run_length", max_run)

    # 4. Duration & metric position
    target_rows = [r for rows in pieces.values() for r in rows if r["deg"] == deg]
    target_durs = [r["dur"] for r in target_rows]
    mean_dur_target = np.mean(target_durs)
    dur_ratio = mean_dur_target / corpus_mean_dur

    offs = [r["offset"] for r in target_rows if r["offset"] is not None]
    frac_offs = [o - int(o) for o in offs]
    near_zero_target = np.mean([f < 0.1 or f > 0.9 for f in frac_offs]) if frac_offs else float("nan")

    all_offs = [r["offset"] for rows in pieces.values() for r in rows if r["offset"] is not None]
    all_frac = [o - int(o) for o in all_offs]
    near_zero_base = np.mean([f < 0.1 or f > 0.9 for f in all_frac]) if all_frac else float("nan")

    print(f"\n4. DURATION & METRIC POSITION")
    print(f"   mean duration: target={mean_dur_target:.1f} ticks  corpus={corpus_mean_dur:.1f}  "
          f"ratio={dur_ratio:.2f}x")
    print(f"   near-beat-boundary proxy: target={100*near_zero_target:.1f}%  "
          f"corpus={100*near_zero_base:.1f}%  [Offset semantics unverified — coarse proxy]")

    add(deg, "4_duration_metric", "mean_dur_ticks_target", round(float(mean_dur_target),2))
    add(deg, "4_duration_metric", "mean_dur_ticks_corpus", round(float(corpus_mean_dur),2))
    add(deg, "4_duration_metric", "dur_ratio", round(float(dur_ratio),3))
    add(deg, "4_duration_metric", "pct_near_beat_target", round(100*near_zero_target,1) if near_zero_target==near_zero_target else "")
    add(deg, "4_duration_metric", "pct_near_beat_corpus", round(100*near_zero_base,1) if near_zero_base==near_zero_base else "")

    # 5. Within-piece position
    norm_positions = []
    for pname in per_piece_count:
        n = len(pieces[pname])
        for r in pieces[pname]:
            if r["deg"] == deg: norm_positions.append(r["idx"] / max(1, n - 1))
    mean_pos = np.mean(norm_positions)
    median_pos = np.median(norm_positions)
    frac_last_q = np.mean([p >= 0.75 for p in norm_positions])

    print(f"\n5. WITHIN-PIECE NORMALIZED POSITION")
    print(f"   mean={mean_pos:.3f}  median={median_pos:.3f}  last-quartile={100*frac_last_q:.1f}%")

    add(deg, "5_position", "mean_norm_pos", round(float(mean_pos),3))
    add(deg, "5_position", "median_norm_pos", round(float(median_pos),3))
    add(deg, "5_position", "pct_last_quartile", round(100*frac_last_q,1))

    # 6. Neighbours
    prev_degs, next_degs = Counter(), Counter()
    for pname in per_piece_count:
        rows = pieces[pname]; n = len(rows)
        for r in rows:
            if r["deg"] != deg: continue
            i = r["idx"]
            if i > 0: prev_degs[rows[i-1]["deg"]] += 1
            if i < n-1: next_degs[rows[i+1]["deg"]] += 1

    print(f"\n6. MOST FREQUENT NEIGHBOURS")
    print(f"   preceding (top 5): {prev_degs.most_common(5)}")
    print(f"   following (top 5): {next_degs.most_common(5)}")

    for rank, (d, c) in enumerate(prev_degs.most_common(5), 1):
        add(deg, "6_neighbours", f"prev_{rank}_degree={d}", c)
    for rank, (d, c) in enumerate(next_degs.most_common(5), 1):
        add(deg, "6_neighbours", f"next_{rank}_degree={d}", c)

    summary_table.append({
        "degree": deg, "name": name, "n_pieces_with": n_pieces_with,
        "pct_pieces": round(100*n_pieces_with/total_pieces,1),
        "median_notes": med_c, "max_notes": max_c,
        "pct_isolated": round(100*n_isolated/n_runs,1),
        "dur_ratio": round(float(dur_ratio),2),
        "mean_norm_pos": round(float(mean_pos),3),
        "top_prev": prev_degs.most_common(1)[0] if prev_degs else None,
        "top_next": next_degs.most_common(1)[0] if next_degs else None,
        "d39_piece_rate": round(100*comp_results.get(39,{}).get("co_piece_rate",float('nan')),1) if deg==37 else "",
        "d39_base_rate": round(100*comp_results.get(39,{}).get("base_rate_piece",float('nan')),1) if deg==37 else "",
    })

# ── critical substitution verdict ─────────────────────────────────────────────
print("\n" + "="*78)
print("CRITICAL TEST: does d=37 substitute for Eviç (d=39), or co-occur with it?")
print("="*78)
d37_comp39 = None
for row in csv_rows:
    if row["degree"]==37 and row["key"]=="comp39_cooccur_piece_pct": d37_comp39 = row["value"]
d39_base = None
for row in csv_rows:
    if row["degree"]==37 and row["key"]=="comp39_base_rate_piece_pct": d39_base = row["value"]
print(f"  Among pieces containing d=37, {d37_comp39}% ALSO contain d=39 (Eviç).")
print(f"  Corpus-wide base rate of d=39 (any piece): {d39_base}%.")
if d37_comp39 is not None and d39_base is not None:
    if d37_comp39 < d39_base - 10:
        verdict = "d=37 tends to SUBSTITUTE for Eviç (co-occurrence below base rate)."
    elif d37_comp39 > d39_base + 10:
        verdict = "d=37 tends to CO-OCCUR with Eviç, not substitute (co-occurrence above base rate)."
    else:
        verdict = "d=37's co-occurrence with Eviç is close to the corpus base rate -- inconclusive."
    print(f"  Verdict: {verdict}")

# ── comparison summary table ──────────────────────────────────────────────────
print("\n" + "="*78)
print("SUMMARY TABLE — d=37 vs. d=39 (Eviç, reference)")
print("="*78)
print(f"{'':>28} {'d=37':>18} {'d=39 (Eviç)':>18}")
fields = [
    ("pieces with degree (%)", "pct_pieces"),
    ("median notes/piece",     "median_notes"),
    ("max notes/piece",        "max_notes"),
    ("isolated occurrences (%)","pct_isolated"),
    ("duration ratio (x corpus)","dur_ratio"),
    ("mean normalized position","mean_norm_pos"),
]
by_deg = {r["degree"]: r for r in summary_table}
for label, key in fields:
    v37 = by_deg[37][key]
    v39 = by_deg[39][key]
    print(f"{label:>28} {str(v37):>18} {str(v39):>18}")
print(f"{'top preceding neighbour':>28} {str(by_deg[37]['top_prev']):>18} {str(by_deg[39]['top_prev']):>18}")
print(f"{'top following neighbour':>28} {str(by_deg[37]['top_next']):>18} {str(by_deg[39]['top_next']):>18}")

# ═════════════════════════════════════════════════════════════════════════════
# Length control: is the d=37 <-> d=39 co-occurrence a piece-length artefact?
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "="*78)
print("LENGTH CONTROL — piece length vs. d=37 presence; d=37<->d=39 controlling for length")
print("="*78)

lengths = {p: len(rows) for p, rows in pieces.items()}
has_37  = {p: int(any(r["deg"] == 37 for r in rows)) for p, rows in pieces.items()}
has_39  = {p: int(any(r["deg"] == 39 for r in rows)) for p, rows in pieces.items()}

len_with37    = [lengths[p] for p in pieces if has_37[p] == 1]
len_without37 = [lengths[p] for p in pieces if has_37[p] == 0]

print(f"\n1. PIECE LENGTH: with d=37 vs without")
print(f"  n(with d=37)    = {len(len_with37)}   mean length = {np.mean(len_with37):.1f}  "
      f"median = {np.median(len_with37):.1f}  sd = {np.std(len_with37, ddof=1):.1f}")
print(f"  n(without d=37) = {len(len_without37)}   mean length = {np.mean(len_without37):.1f}  "
      f"median = {np.median(len_without37):.1f}  sd = {np.std(len_without37, ddof=1):.1f}")

t_len, p_t_len = ss.ttest_ind(len_with37, len_without37, equal_var=False)
u_len, p_u_len = ss.mannwhitneyu(len_with37, len_without37, alternative='two-sided')
print(f"\n  Welch t-test:      t={t_len:.3f}, p={p_t_len:.4f}")
print(f"  Mann-Whitney U:    U={u_len:.1f}, p={p_u_len:.4f}")

len_ratio = np.mean(len_with37) / np.mean(len_without37)
print(f"  Ratio (with/without): {len_ratio:.2f}x")
length_confound = p_t_len < .05
print(f"  => {'Piece length IS a plausible confound.' if length_confound else 'No significant piece-length difference; length confound unlikely to be large.'}")

print(f"\n2. Does d=37 <-> d=39 (Eviç) association survive controlling for length?")

names_p = list(pieces.keys())
X_has37 = np.array([has_37[p] for p in names_p], dtype=float)
X_len   = np.array([lengths[p] for p in names_p], dtype=float)
y_has39 = np.array([has_39[p] for p in names_p], dtype=float)

X_len_z = (X_len - X_len.mean()) / X_len.std()
X_logit = sm.add_constant(np.column_stack([X_has37, X_len_z]))
model = sm.Logit(y_has39, X_logit).fit(disp=0)
print(model.summary2().tables[1].to_string())

b_const, b_has37, b_len = model.params
se_has37 = model.bse[1]
p_has37  = model.pvalues[1]
or_has37 = np.exp(b_has37)
ci_lo, ci_hi = np.exp(model.conf_int()[1])

print(f"\n  Odds ratio for d=37 presence predicting d=39 presence (controlling for length):")
print(f"    OR = {or_has37:.2f}   95% CI [{ci_lo:.2f}, {ci_hi:.2f}]   "
      f"beta={b_has37:.3f}  SE={se_has37:.3f}  p={p_has37:.4f}")

raw_rate_with37    = np.mean([has_39[p] for p in names_p if has_37[p]==1])
raw_rate_without37 = np.mean([has_39[p] for p in names_p if has_37[p]==0])
print(f"\n  Raw (unadjusted) rates for reference:")
print(f"    P(d=39 | d=37 present)  = {100*raw_rate_with37:.1f}%")
print(f"    P(d=39 | d=37 absent)   = {100*raw_rate_without37:.1f}%")

if p_has37 < .05:
    verdict_len = ("The d=37 <-> d=39 co-occurrence is SIGNIFICANT even after controlling "
                   "for piece length -- not just an artefact of longer pieces containing more "
                   "rare degrees in general.")
else:
    verdict_len = ("The d=37 <-> d=39 co-occurrence is NOT significant once piece length is "
                   "controlled for -- consistent with (at least partly) a length confound.")
print(f"\n  VERDICT: {verdict_len}")

for key, val in [
    ("n_with_d37", len(len_with37)), ("mean_len_with_d37", round(float(np.mean(len_with37)),2)),
    ("n_without_d37", len(len_without37)), ("mean_len_without_d37", round(float(np.mean(len_without37)),2)),
    ("length_ttest_t", round(float(t_len),4)), ("length_ttest_p", round(float(p_t_len),6)),
    ("length_mannwhitney_U", round(float(u_len),2)), ("length_mannwhitney_p", round(float(p_u_len),6)),
    ("logit_OR_d37_on_d39", round(float(or_has37),4)),
    ("logit_OR_ci_lo", round(float(ci_lo),4)), ("logit_OR_ci_hi", round(float(ci_hi),4)),
    ("logit_p_d37", round(float(p_has37),6)),
    ("raw_rate_d39_given_d37", round(100*float(raw_rate_with37),2)),
    ("raw_rate_d39_given_no_d37", round(100*float(raw_rate_without37),2)),
]:
    add(37, "7_length_control", key, val)

out = RESULTS / "d37_context.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["maqam","degree","section","key","value"])
    w.writeheader()
    w.writerows(csv_rows)
print(f"\n-> Saved: {out}  ({len(csv_rows)} rows)")
