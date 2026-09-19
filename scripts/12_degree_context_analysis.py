"""Co-occurrence tests for high-ΔIC degrees (Nihâvend d=36 hisar, d=17 segâh; Uşşak d=8 reference) against predicted companion degrees, plus segah/kürdi prevalence across the d=4-9 region."""
import csv
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

SYMBTR_DIR = Path("/Users/ugurozalp/Downloads/SymbTr-master/txt")   # local SymbTr checkout; edit for your setup
RESULTS    = Path(__file__).resolve().parent.parent / "results"
WHOLE_TICKS = 384

WINDOW = 16
CLUSTER_GAP = 3   # max index-gap to count as "same run" in temporal clustering

LABELS = {"ussak": "Uşşak", "huseyni": "Hüseyni", "nihavent": "Nihâvend"}

TARGETS = [
    ("nihavent", 36, "hisar",  [48]),
    ("nihavent", 17, "segâh", [13, 14]),
    ("ussak",    8,  "segah (ref)", [13]),
]

def _int(s):
    try: return int(s)
    except: return None

def _float(s):
    try: return float(s)
    except: return None

def parse_piece(path):
    """Returns list of dicts: koma, dur_ticks, offset (float or None), in note order."""
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
    """Returns {piece_name: [ {koma,dur,offset,idx,deg} ... ]}, karar per piece."""
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

print("Loading corpora …")
CORPUS = {}
# "huseyni" is loaded too (in addition to the TARGETS makams nihavent/ussak)
# because segah_prevalence() below needs all three makams.
for m in ["nihavent", "ussak", "huseyni"]:
    CORPUS[m] = load_corpus(m)
    n_notes = sum(len(v) for v in CORPUS[m].values())
    print(f"  {m:10s}  {len(CORPUS[m])} pieces, {n_notes} notes")

csv_rows = []

def add(maqam, deg, section, key, value):
    csv_rows.append({"maqam": maqam, "degree": deg, "section": section,
                     "key": key, "value": value})

for maqam, deg, name, companions in TARGETS:
    pieces = CORPUS[maqam]
    total_pieces = len(pieces)
    all_durs = [r["dur"] for p in pieces.values() for r in p]
    corpus_mean_dur = np.mean(all_durs)

    print("\n" + "="*78)
    print(f"{maqam.upper()}  d={deg}  ({name})   companions tested: {companions}")
    print("="*78)

    # ── 1. Distribution ──────────────────────────────────────────────────────
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
    print(f"   total occurrences in corpus: {total_occ}")
    concentrated = "concentrated in FEW pieces" if n_pieces_with <= 5 else \
                   ("spread THIN across many pieces" if med_c <= 2 else "moderate")
    print(f"   pattern: {concentrated}")

    add(maqam, deg, "1_distribution", "n_pieces_with", n_pieces_with)
    add(maqam, deg, "1_distribution", "total_pieces", total_pieces)
    add(maqam, deg, "1_distribution", "pct_pieces_with", round(100*n_pieces_with/total_pieces, 2))
    add(maqam, deg, "1_distribution", "median_notes_per_piece", med_c)
    add(maqam, deg, "1_distribution", "max_notes_per_piece", max_c)
    add(maqam, deg, "1_distribution", "total_occurrences", total_occ)

    # ── 2. Tetrachord signature ─────────────────────────────────────────────
    print(f"\n2. TETRACHORD SIGNATURE")

    # base rate of each companion degree across ALL pieces (piece-level presence)
    for comp in companions:
        base_pieces_with_comp = sum(1 for rows in pieces.values()
                                     if any(r["deg"] == comp for r in rows))
        base_rate_piece = base_pieces_with_comp / total_pieces

        # piece-level co-occurrence: among pieces WITH target deg, fraction that ALSO have comp
        co_piece = sum(1 for pname in per_piece_count
                       if any(r["deg"] == comp for r in pieces[pname]))
        co_piece_rate = co_piece / n_pieces_with if n_pieces_with else float("nan")

        # window-level: for each occurrence of target deg, does comp appear in +/-16 window?
        n_occ_with_window_comp = 0
        n_occ_total = 0
        for pname, rows in pieces.items():
            degs = [r["deg"] for r in rows]
            n = len(degs)
            for i, d in enumerate(degs):
                if d != deg: continue
                n_occ_total += 1
                lo, hi = max(0, i - WINDOW), min(n, i + WINDOW + 1)
                if comp in degs[lo:i] + degs[i+1:hi]:
                    n_occ_with_window_comp += 1
        window_rate = n_occ_with_window_comp / n_occ_total if n_occ_total else float("nan")

        # base rate at window level: for ALL notes in corpus (any degree), what fraction
        # have comp within +/-16 window? (fair baseline for comparison)
        n_base_total = 0
        n_base_with_comp = 0
        for pname, rows in pieces.items():
            degs = [r["deg"] for r in rows]
            n = len(degs)
            for i in range(n):
                n_base_total += 1
                lo, hi = max(0, i - WINDOW), min(n, i + WINDOW + 1)
                if comp in degs[lo:i] + degs[i+1:hi]:
                    n_base_with_comp += 1
        window_base_rate = n_base_with_comp / n_base_total if n_base_total else float("nan")

        lift = window_rate / window_base_rate if window_base_rate > 0 else float("nan")

        print(f"   companion d={comp}:")
        print(f"     corpus base rate (piece-level presence)  = {100*base_rate_piece:.1f}%")
        print(f"     co-occurrence rate (piece-level, target-> comp) = {100*co_piece_rate:.1f}%  "
              f"({co_piece}/{n_pieces_with} pieces)")
        print(f"     window co-occurrence rate (±{WINDOW} notes)     = {100*window_rate:.1f}%  "
              f"({n_occ_with_window_comp}/{n_occ_total} occurrences)")
        print(f"     corpus-wide window base rate (any position)    = {100*window_base_rate:.1f}%")
        print(f"     lift (window_rate / window_base_rate)          = {lift:.2f}x")

        add(maqam, deg, "2_tetrachord", f"comp{comp}_base_rate_piece_pct", round(100*base_rate_piece,2))
        add(maqam, deg, "2_tetrachord", f"comp{comp}_cooccur_piece_pct", round(100*co_piece_rate,2))
        add(maqam, deg, "2_tetrachord", f"comp{comp}_window_rate_pct", round(100*window_rate,2))
        add(maqam, deg, "2_tetrachord", f"comp{comp}_window_base_rate_pct", round(100*window_base_rate,2))
        add(maqam, deg, "2_tetrachord", f"comp{comp}_lift", round(lift,3) if lift==lift else "")

    # ── 3. Temporal clustering ──────────────────────────────────────────────
    print(f"\n3. TEMPORAL CLUSTERING (gap<= {CLUSTER_GAP} notes = same run)")
    run_lengths = []
    for pname in per_piece_count:
        idxs = sorted(r["idx"] for r in pieces[pname] if r["deg"] == deg)
        run = [idxs[0]]
        for prev, cur in zip(idxs, idxs[1:]):
            if cur - prev <= CLUSTER_GAP:
                run.append(cur)
            else:
                run_lengths.append(len(run))
                run = [cur]
        run_lengths.append(len(run))

    n_runs = len(run_lengths)
    n_isolated = sum(1 for r in run_lengths if r == 1)
    mean_run = np.mean(run_lengths) if run_lengths else float("nan")
    max_run = max(run_lengths) if run_lengths else 0
    print(f"   n runs = {n_runs}, isolated (len=1) = {n_isolated} "
          f"({100*n_isolated/n_runs:.1f}%), mean run length = {mean_run:.2f}, max = {max_run}")
    pattern = "ISOLATED (mostly single notes)" if n_isolated/n_runs > 0.7 else "CLUSTERED (occurs in bursts/passages)"
    print(f"   pattern: {pattern}")

    add(maqam, deg, "3_clustering", "n_runs", n_runs)
    add(maqam, deg, "3_clustering", "pct_isolated", round(100*n_isolated/n_runs,1))
    add(maqam, deg, "3_clustering", "mean_run_length", round(float(mean_run),3))
    add(maqam, deg, "3_clustering", "max_run_length", max_run)

    # ── 4. Duration & metric position ───────────────────────────────────────
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
    print(f"   mean duration (ticks): target={mean_dur_target:.1f}  corpus={corpus_mean_dur:.1f}  "
          f"ratio={dur_ratio:.2f}x")
    print(f"   near-beat-boundary proxy (frac(Offset)<0.1 or >0.9): "
          f"target={100*near_zero_target:.1f}%  corpus={100*near_zero_base:.1f}%  "
          f"[NOTE: Offset's exact usul semantics unverified — coarse proxy only]")

    add(maqam, deg, "4_duration_metric", "mean_dur_ticks_target", round(float(mean_dur_target),2))
    add(maqam, deg, "4_duration_metric", "mean_dur_ticks_corpus", round(float(corpus_mean_dur),2))
    add(maqam, deg, "4_duration_metric", "dur_ratio", round(float(dur_ratio),3))
    add(maqam, deg, "4_duration_metric", "pct_near_beat_boundary_target", round(100*near_zero_target,1) if near_zero_target==near_zero_target else "")
    add(maqam, deg, "4_duration_metric", "pct_near_beat_boundary_corpus", round(100*near_zero_base,1) if near_zero_base==near_zero_base else "")

    # ── 5. Within-piece normalized position ─────────────────────────────────
    norm_positions = []
    for pname in per_piece_count:
        n = len(pieces[pname])
        for r in pieces[pname]:
            if r["deg"] == deg:
                norm_positions.append(r["idx"] / max(1, n - 1))
    mean_pos = np.mean(norm_positions)
    median_pos = np.median(norm_positions)
    frac_last_quartile = np.mean([p >= 0.75 for p in norm_positions])

    print(f"\n5. WITHIN-PIECE NORMALIZED POSITION")
    print(f"   mean={mean_pos:.3f}  median={median_pos:.3f}  "
          f"fraction in last quartile (>=0.75)={100*frac_last_quartile:.1f}%")
    ending_bias = "concentrated toward the END" if frac_last_quartile > 0.35 else "roughly uniform / not end-biased"
    print(f"   pattern: {ending_bias}")

    add(maqam, deg, "5_position", "mean_norm_pos", round(float(mean_pos),3))
    add(maqam, deg, "5_position", "median_norm_pos", round(float(median_pos),3))
    add(maqam, deg, "5_position", "pct_last_quartile", round(100*frac_last_quartile,1))

    # ── 6. Most frequent neighbours ──────────────────────────────────────────
    prev_degs, next_degs = Counter(), Counter()
    for pname in per_piece_count:
        rows = pieces[pname]
        n = len(rows)
        for r in rows:
            if r["deg"] != deg: continue
            i = r["idx"]
            if i > 0: prev_degs[rows[i-1]["deg"]] += 1
            if i < n-1: next_degs[rows[i+1]["deg"]] += 1

    print(f"\n6. MOST FREQUENT NEIGHBOURS")
    print(f"   preceding (top 5): {prev_degs.most_common(5)}")
    print(f"   following (top 5): {next_degs.most_common(5)}")

    for rank, (d, c) in enumerate(prev_degs.most_common(5), 1):
        add(maqam, deg, "6_neighbours", f"prev_{rank}_degree={d}", c)
    for rank, (d, c) in enumerate(next_degs.most_common(5), 1):
        add(maqam, deg, "6_neighbours", f"next_{rank}_degree={d}", c)

out = RESULTS / "degree_context_analysis.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["maqam","degree","section","key","value"])
    w.writeheader()
    w.writerows(csv_rows)
print(f"\n\n-> Saved: {out}  ({len(csv_rows)} rows)")

# ═════════════════════════════════════════════════════════════════════════════
# Segah prevalence — supports the paper's claim that segah (d=8) is only the
# MAJORITY choice in a minority of Uşşak pieces (most 2nd-degree occurrences
# in Uşşak are kürdi, d=7), i.e. the neutral-second choice is corpus-variable
# rather than fixed, which matters for interpreting the d=7/8 conflation result.
# ═════════════════════════════════════════════════════════════════════════════

def segah_prevalence():
    print("\n" + "="*70)
    print("SEGAH PREVALENCE  (d ∈ {4,5,6,7,8,9}, karar-relative mod 53)")
    print("="*70)
    DEGREES = [4, 5, 6, 7, 8, 9]
    rows = []
    all_makams = ["ussak", "huseyni", "nihavent"]

    for m in all_makams:
        pieces = CORPUS[m]
        piece_degs = defaultdict(set)
        deg_total  = defaultdict(int)
        for pname, prows in pieces.items():
            for r in prows:
                piece_degs[pname].add(r["deg"])
                deg_total[r["deg"]] += 1
        n_pieces = len(pieces)
        n_notes  = sum(len(v) for v in pieces.values())

        print(f"\n  {LABELS[m]}  (n_pieces = {n_pieces}, n_notes_total = {n_notes})")
        print(f"  {'d':>4}  {'n pieces with d':>16}  {'% pieces':>10}  {'total notes':>12}")
        print("  " + "─"*48)
        for d in DEGREES:
            n_with = sum(1 for pname in pieces if d in piece_degs[pname])
            pct    = n_with / n_pieces * 100
            flag   = "  ← segah" if d==8 and m!="nihavent" else \
                     "  ← kürdi" if d==7 and m!="nihavent" else ""
            print(f"  {d:4d}  {n_with:16d}  {pct:10.1f}%  {deg_total[d]:12d}{flag}")
            rows.append(dict(maqam=LABELS[m], degree=d, n_pieces_total=n_pieces,
                             n_pieces_with_d=n_with, pct_pieces=round(pct,2),
                             total_notes=deg_total[d]))

    # Uşşak pieces without d=8 — most common degree in 4–9 region
    print(f"\n  ─── Uşşak pieces WITHOUT d = 8 ───")
    pieces_u = CORPUS["ussak"]
    piece_degs_u  = defaultdict(set)
    piece_deg_cnt = defaultdict(lambda: defaultdict(int))
    for pname, prows in pieces_u.items():
        for r in prows:
            piece_degs_u[pname].add(r["deg"])
            piece_deg_cnt[pname][r["deg"]] += 1

    no_d8 = [pname for pname in pieces_u if 8 not in piece_degs_u[pname]]
    print(f"  Pieces without d=8:  {len(no_d8)} / {len(pieces_u)}"
          f"  ({len(no_d8)/len(pieces_u)*100:.1f}%)")

    agg = defaultdict(int)
    for pname in no_d8:
        for d in DEGREES:
            agg[d] += piece_deg_cnt[pname][d]

    print(f"\n  Notes in d=4–9 region  (only pieces without d=8):")
    print(f"  {'d':>4}  {'notes':>10}  {'%':>8}")
    total_246 = sum(agg[d] for d in DEGREES)
    for d in DEGREES:
        pct = agg[d] / total_246 * 100 if total_246 else 0
        marker = "  ← most common" if d == max(DEGREES, key=lambda x: agg[x]) else ""
        print(f"  {d:4d}  {agg[d]:10d}  {pct:7.1f}%{marker}")

    dom_d = max(DEGREES, key=lambda x: agg[x])
    print(f"\n  → Most common 2nd-degree in Uşşak pieces without d=8: d = {dom_d}")

    # Segah (d=8) vs Kürdi (d=7) note-count summary
    print(f"\n  ─── Segah (d=8) vs Kürdi (d=7) summary ───")
    print(f"  {'Maqam':12s}  {'d=7 notes':>10}  {'d=8 notes':>10}  {'ratio d7/d8':>12}")
    print("  " + "─"*50)
    for m in ["ussak", "huseyni"]:
        cnt = defaultdict(int)
        for prows in CORPUS[m].values():
            for r in prows: cnt[r["deg"]] += 1
        r7, r8 = cnt[7], cnt[8]
        ratio  = r7/r8 if r8 > 0 else float("inf")
        print(f"  {LABELS[m]:12s}  {r7:10d}  {r8:10d}  {ratio:12.2f}×")

    out2 = RESULTS / "segah_prevalence.csv"
    with open(out2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["maqam","degree","n_pieces_total",
                                          "n_pieces_with_d","pct_pieces","total_notes"])
        w.writeheader(); w.writerows(rows)
    print(f"\n  → Saved: {out2}")

segah_prevalence()
