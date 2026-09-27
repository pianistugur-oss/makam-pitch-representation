"""Counterfactual corpora with Uşşak/Hüseyni's d=7,8 region uniformly merged to segah or to kürdi (octave-preserving), retrained from scratch, and compared against the observed corpus."""

import argparse, csv, os, sqlite3, subprocess, sys, time
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from scipy import stats as ss

BASE       = Path("/Users/ugurozalp/makam_beklenti")          # local IDyOM output location; edit for your setup
OUTDIR     = BASE / "data" / "idyom_output"
SCRIPTS    = BASE / "scripts"
RESULTS    = Path(__file__).resolve().parent.parent / "results"
SQLITE_LIB = "/opt/homebrew/opt/sqlite/lib"                    # macOS IDyOM sqlite lib; edit for your setup

LABELS = {"ussak": "Uşşak", "huseyni": "Hüseyni", "nihavent": "Nihâvend"}
MICRO  = ["ussak", "huseyni"]

ORIG_DATASET = {"ussak": 200, "huseyni": 202}
# (makam, target_deg) -> (dataset_id, subdir, label). target_deg=8 -> merge
# everything in the d=7/8 region to segah; target_deg=7 -> merge to kürdi.
VARIANTS = {
    ("ussak",   8): (210, "variant_210", "A (all -> segah, koma=313)"),
    ("ussak",   7): (211, "variant_211", "B (all -> kürdi, koma=312)"),
    ("huseyni", 8): (212, "variant_212", "A (all -> segah, koma=313)"),
    ("huseyni", 7): (213, "variant_213", "B (all -> kürdi, koma=312)"),
}

def koma_to_tet(k): return round(int(k) * 12 / 53)

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — build the 4 respelling-variant databases (MERGE, octave-preserving)
# ═════════════════════════════════════════════════════════════════════════════
# new_koma = (koma - deg) + target_deg, where deg = (koma - karar) % 53.
# This assigns EVERY note in the d=7-or-8 region to a single koma value
# (313 for "all segah", 312 for "all kürdi") while preserving the octave the
# note actually occurred in -- NOT a fixed +1/-1 shift, which would instead
# push originally-segah notes (koma 313) up to koma 314 (out of the region).
# 312 and 313 map to the same 12-TET class (MIDI 71), so the 12-TET arm is
# provably unchanged by this operation (verified below) and its existing
# .dat output is reused rather than re-run.

def get_karar_map(makam):
    db = OUTDIR / makam / "koma" / "idyom.db"
    with sqlite3.connect(db) as c:
        rows = c.execute(
            "SELECT COMPOSITION_ID, CPITCH FROM mtp_event "
            "WHERE DATASET_ID=? ORDER BY COMPOSITION_ID, ONSET",
            (ORIG_DATASET[makam],)).fetchall()
    last = {}
    for cid, cp in rows: last[cid] = cp
    return last

def load_all_events(makam):
    db = OUTDIR / makam / "koma" / "idyom.db"
    with sqlite3.connect(db) as c:
        c.row_factory = sqlite3.Row
        cols = [r[1] for r in c.execute("PRAGMA table_info(mtp_event)")]
        rows = c.execute(
            "SELECT * FROM mtp_event WHERE DATASET_ID=? ORDER BY EVENT_ID",
            (ORIG_DATASET[makam],)).fetchall()
    return cols, [dict(r) for r in rows]

def load_compositions(makam):
    db = OUTDIR / makam / "koma" / "idyom.db"
    with sqlite3.connect(db) as c:
        c.row_factory = sqlite3.Row
        rows = c.execute(
            "SELECT * FROM mtp_composition WHERE DATASET_ID=?",
            (ORIG_DATASET[makam],)).fetchall()
    return [dict(r) for r in rows]

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS mtp_dataset (
    DATASET_ID INTEGER PRIMARY KEY, DESCRIPTION VARCHAR(255),
    TIMEBASE INTEGER, MIDC INTEGER
);
CREATE TABLE IF NOT EXISTS mtp_composition (
    COMPOSITION_ID INTEGER, DATASET_ID INTEGER, TIMEBASE INTEGER,
    DESCRIPTION VARCHAR(255), PRIMARY KEY (DATASET_ID, COMPOSITION_ID)
);
CREATE TABLE IF NOT EXISTS mtp_event (
    EVENT_ID INTEGER PRIMARY KEY, COMPOSITION_ID INTEGER, DATASET_ID INTEGER,
    ONSET INTEGER, CPITCH INTEGER, MPITCH INTEGER, ACCIDENTAL INTEGER,
    DUR INTEGER, DELTAST INTEGER, BIOI INTEGER, KEYSIG INTEGER, MODE INTEGER,
    BARLENGTH INTEGER, PULSES INTEGER, PHRASE INTEGER, TEMPO INTEGER,
    DYN INTEGER, ORNAMENT INTEGER, COMMA INTEGER, ARTICULATION INTEGER,
    VERTINT12 INTEGER, VOICE INTEGER
);
"""

def write_variant_db(makam, new_dataset_id, description, events, compositions, cols):
    out_dir = OUTDIR / makam / f"variant_{new_dataset_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "idyom.db"
    if db_path.exists(): db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_SQL)
    conn.execute("INSERT OR REPLACE INTO mtp_dataset VALUES (?,?,?,?)",
                 (new_dataset_id, description, 96, 265))
    for comp in compositions:
        conn.execute("INSERT OR REPLACE INTO mtp_composition VALUES (?,?,?,?)",
                     (comp["COMPOSITION_ID"], new_dataset_id, comp["TIMEBASE"], comp["DESCRIPTION"]))
    placeholders = ",".join("?" * len(cols))
    rows = [[ev[c] if c != "DATASET_ID" else new_dataset_id for c in cols] for ev in events]
    conn.executemany(f"INSERT INTO mtp_event VALUES ({placeholders})", rows)
    conn.commit(); conn.close()
    return db_path

def build_variant_databases():
    print("STEP 1 — Building respelling-variant databases (merge, octave-preserving) …")
    for makam in ["ussak", "huseyni"]:
        karar = get_karar_map(makam)
        cols, events = load_all_events(makam)
        compositions = load_compositions(makam)

        for target_deg in (8, 7):
            new_id, subdir, label = VARIANTS[(makam, target_deg)]
            new_events = []
            n_changed = 0
            tet_mismatch = 0
            for ev in events:
                cp = ev["CPITCH"]
                k = karar.get(ev["COMPOSITION_ID"])
                deg = (cp - k) % 53 if k is not None else None
                new_ev = dict(ev)
                if deg in (7, 8):
                    new_cp = (cp - deg) + target_deg
                    new_ev["CPITCH"] = new_cp
                    n_changed += 1
                    if koma_to_tet(new_cp) != koma_to_tet(cp):
                        tet_mismatch += 1
                new_events.append(new_ev)

            print(f"  {makam} target_deg={target_deg} ({label}) dataset {new_id}: "
                  f"{n_changed} events merged, 12-TET mismatches={tet_mismatch} "
                  f"({'OK' if tet_mismatch==0 else 'BUG'})")
            write_variant_db(makam, new_id, f"{makam}_respell_{label}", new_events, compositions, cols)

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — run IDyOM on each of the 4 respelling-variant databases
# (12-TET arm is unchanged by construction -- reuse the existing tet12 output)
# ═════════════════════════════════════════════════════════════════════════════

def lisp_for(makam, dsid, subdir, cv_fold=10):
    db_path = OUTDIR / makam / subdir / "idyom.db"
    output_dir = OUTDIR / makam / subdir
    lisp_path = SCRIPTS / f"idyom_{makam}_{subdir}.lisp"
    lisp = f""";; Auto-generated
(ql:quickload "clsql-sqlite3" :silent t)
(ql:quickload "idyom" :silent t)
(clsql:connect (list "{db_path}")
               :database-type :sqlite3
               :if-exists :old)
(idyom:idyom {dsid}
             '(cpitch)
             '(cpint cpitch)
             :models :both+
             :k {cv_fold}
             :detail 3
             :output-path "{output_dir}/"
             :overwrite t)
(format t "~%Done: {makam}/{subdir} (dataset {dsid})~%")
(quit)
"""
    lisp_path.write_text(lisp)
    return lisp_path

def run_one(makam, dsid, subdir):
    lisp_path = lisp_for(makam, dsid, subdir)
    env = os.environ.copy()
    env["DYLD_LIBRARY_PATH"] = SQLITE_LIB
    t0 = time.time()
    print(f"--- {makam}/{subdir} (dataset {dsid}) starting ---", flush=True)
    result = subprocess.run(
        ["sbcl", "--noinform", "--load", str(lisp_path)],
        env=env, capture_output=True, text=True,
    )
    dt = time.time() - t0
    ok = result.returncode == 0
    dat_files = list((OUTDIR / makam / subdir).glob("*.dat"))
    print(f"--- {makam}/{subdir} finished in {dt:.1f}s  ok={ok}  dat_files={len(dat_files)} ---", flush=True)
    if not ok:
        print("STDERR TAIL:", result.stderr[-3000:], file=sys.stderr, flush=True)
    return ok, dt

def run_idyom_on_variants():
    print("\nSTEP 2 — Running IDyOM on the 4 respelling-variant databases (~15-20 min) …")
    results = []
    for (makam, target_deg), (dsid, subdir, label) in VARIANTS.items():
        ok, dt = run_one(makam, dsid, subdir)
        results.append((makam, dsid, subdir, ok, dt))
    n_ok = sum(1 for r in results if r[3])
    print(f"\n=== IDyOM runs: {n_ok}/{len(results)} succeeded ===")
    return results

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — analyze: compare each counterfactual corpus against the observed one
# ═════════════════════════════════════════════════════════════════════════════

def read_dat_full(path):
    with open(path) as f: hdr = f.readline().split()
    mid_i, nid_i = hdr.index("melody.id"), hdr.index("note.id")
    cp_i, ic_i   = hdr.index("cpitch"), hdr.index("cpitch.ic")
    out = {}
    with open(path) as f:
        f.readline()
        for line in f:
            pts = line.split()
            if pts[ic_i] == "NA": continue
            out[(int(pts[mid_i]), int(pts[nid_i]))] = (int(pts[cp_i]), float(pts[ic_i]))
    return out

def get_dat(dir_path):
    files = sorted(dir_path.glob("*.dat"))
    if not files: raise FileNotFoundError(f"No .dat in {dir_path}")
    return files[0]

def piece_level_mean_dic(dic_by_key):
    per_mid = defaultdict(list)
    for (mid, nid), v in dic_by_key.items(): per_mid[mid].append(v)
    return {mid: np.mean(v) for mid, v in per_mid.items()}

def cohens_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na-1)*np.var(a,ddof=1)+(nb-1)*np.var(b,ddof=1))/(na+nb-2))
    return (np.mean(a)-np.mean(b))/sp if sp > 0 else 0.0

def fmt_p(p): return "< .001" if p < .001 else f"= {p:.4f}"

def analyze():
    print("\nSTEP 3 — Analyzing counterfactual corpora vs. observed …")
    orig_tet_dat, orig_koma_dat = {}, {}
    orig_dic_note, orig_dic_piece = {}, {}
    karar_map = {}

    for m in ["ussak", "huseyni", "nihavent"]:
        dk = read_dat_full(get_dat(OUTDIR / m / "koma"))
        dt = read_dat_full(get_dat(OUTDIR / m / "tet12"))
        common = sorted(set(dk) & set(dt))
        dic = {k: dk[k][1] - dt[k][1] for k in common}
        orig_dic_note[m] = dic
        orig_dic_piece[m] = piece_level_mean_dic(dic)
        orig_tet_dat[m] = dt
        orig_koma_dat[m] = dk
        karar_map[m] = get_karar_map(m) if m in ORIG_DATASET else None
        print(f"  {LABELS[m]:12s}  {len(common)} matched notes, {len(orig_dic_piece[m])} pieces")

    csv_rows = []
    variant_results = {}

    for (makam, target_deg), (dsid, subdir, vname) in VARIANTS.items():
        dv = read_dat_full(get_dat(OUTDIR / makam / subdir))
        dt = orig_tet_dat[makam]
        common = sorted(set(dv) & set(dt))
        dic_variant = {k: dv[k][1] - dt[k][1] for k in common}

        karar = karar_map[makam]
        dk_orig = orig_koma_dat[makam]
        deg_of = {}
        for k in common:
            cp_orig, _ = dk_orig.get(k, (None, None))
            if cp_orig is None: continue
            comp_id = k[0] - 1
            kar = karar.get(comp_id)
            if kar is None: continue
            deg_of[k] = (cp_orig - kar) % 53

        d7_keys = [k for k in common if deg_of.get(k) == 7]
        d8_keys = [k for k in common if deg_of.get(k) == 8]

        orig_dic = orig_dic_note[makam]
        piece_variant = piece_level_mean_dic(dic_variant)
        piece_orig    = orig_dic_piece[makam]
        common_pieces = sorted(set(piece_variant) & set(piece_orig))
        pv = np.array([piece_variant[p] for p in common_pieces])
        po = np.array([piece_orig[p]    for p in common_pieces])
        diff = pv - po

        W, p_w = ss.wilcoxon(diff, alternative="two-sided") if np.any(diff != 0) else (np.nan, np.nan)

        d7_orig_mean = np.mean([orig_dic[k] for k in d7_keys]) if d7_keys else np.nan
        d8_orig_mean = np.mean([orig_dic[k] for k in d8_keys]) if d8_keys else np.nan
        d7_var_mean  = np.mean([dic_variant[k] for k in d7_keys]) if d7_keys else np.nan
        d8_var_mean  = np.mean([dic_variant[k] for k in d8_keys]) if d8_keys else np.nan

        res = {
            "n_pieces": len(common_pieces), "n_notes": len(common),
            "orig_piece_mean": float(po.mean()), "variant_piece_mean": float(pv.mean()),
            "diff_mean": float(diff.mean()), "diff_med": float(np.median(diff)),
            "wilcoxon_W": float(W) if W==W else None, "wilcoxon_p": float(p_w) if p_w==p_w else None,
            "d7_orig": float(d7_orig_mean), "d7_variant": float(d7_var_mean),
            "d8_orig": float(d8_orig_mean), "d8_variant": float(d8_var_mean),
            "pv": pv,
        }
        variant_results[(makam, target_deg)] = res

        print(f"\n  {LABELS[makam]}  variant {vname}  (dataset {dsid})")
        print(f"    n pieces={res['n_pieces']}  n notes={res['n_notes']}")
        print(f"    piece ΔIC: orig={res['orig_piece_mean']:+.5f}  "
              f"variant={res['variant_piece_mean']:+.5f}  diff={res['diff_mean']:+.5f}")
        if W == W:
            print(f"    Wilcoxon W={W:.1f}, p {fmt_p(p_w)}")
        print(f"    d=7 region: orig={res['d7_orig']:+.4f} -> variant={res['d7_variant']:+.4f}")
        print(f"    d=8 region: orig={res['d8_orig']:+.4f} -> variant={res['d8_variant']:+.4f}")

        for key in ["n_pieces","n_notes","orig_piece_mean","variant_piece_mean",
                    "diff_mean","diff_med","wilcoxon_W","wilcoxon_p",
                    "d7_orig","d7_variant","d8_orig","d8_variant"]:
            v = res[key]
            csv_rows.append({"maqam":LABELS[makam],"variant":vname,"key":key,
                             "value": "" if v is None else round(float(v),6)})

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 4 — H(comma | MIDI 71) and MIDI-71 class composition, before/after
    # merge. Computed directly from note frequencies in the already-produced
    # .dat output (original koma arm + each variant's koma arm) — no retraining.
    # ═════════════════════════════════════════════════════════════════════════

    def entropy_bits(counts):
        total = sum(counts)
        if total == 0: return float("nan")
        return -sum((c/total)*np.log2(c/total) for c in counts if c > 0)

    print("\n  H(comma | MIDI 71) and MIDI-71 composition, before/after merge:")
    for m in ["ussak", "huseyni"]:
        orig_komas = [cp for (cp, _ic) in orig_koma_dat[m].values() if koma_to_tet(cp) == 71]
        orig_counts = Counter(orig_komas)
        H_orig = entropy_bits(list(orig_counts.values()))
        comp_str = ", ".join(f"{k}:{c}" for k, c in sorted(orig_counts.items()))
        print(f"    {LABELS[m]:10s} original         H={H_orig:.4f}  composition={{{comp_str}}}")
        csv_rows.append({"maqam": LABELS[m], "variant": "original",
                         "key": "H_comma_given_tet71", "value": round(H_orig, 6)})
        for koma_val, cnt in sorted(orig_counts.items()):
            csv_rows.append({"maqam": LABELS[m], "variant": "original",
                             "key": f"tet71_composition_koma={koma_val}", "value": cnt})

        for target_deg, vname in [(8, "A (all -> segah, koma=313)"), (7, "B (all -> kürdi, koma=312)")]:
            dsid, subdir, _ = VARIANTS[(m, target_deg)]
            dv = read_dat_full(get_dat(OUTDIR / m / subdir))
            var_komas = [cp for (cp, _ic) in dv.values() if koma_to_tet(cp) == 71]
            var_counts = Counter(var_komas)
            H_var = entropy_bits(list(var_counts.values()))
            comp_str = ", ".join(f"{k}:{c}" for k, c in sorted(var_counts.items()))
            print(f"    {LABELS[m]:10s} {vname:28s} H={H_var:.4f}  composition={{{comp_str}}}")
            csv_rows.append({"maqam": LABELS[m], "variant": vname,
                             "key": "H_comma_given_tet71", "value": round(H_var, 6)})
            for koma_val, cnt in sorted(var_counts.items()):
                csv_rows.append({"maqam": LABELS[m], "variant": vname,
                                 "key": f"tet71_composition_koma={koma_val}", "value": cnt})

    print("\n  Micro-vs-control comparison (piece-level ΔIC under each variant):")
    ctrl_arr = np.array(list(orig_dic_piece["nihavent"].values()))
    for target_deg, vname in [(8, "A (all -> segah, koma=313)"), (7, "B (all -> kürdi, koma=312)")]:
        micro_arr = np.concatenate([variant_results[(m, target_deg)]["pv"] for m in MICRO])
        U, p_u = ss.mannwhitneyu(micro_arr, ctrl_arr, alternative="two-sided")
        d_u = cohens_d(micro_arr, ctrl_arr)
        print(f"  Variant {vname}: U={U:.1f}, p {fmt_p(p_u)}, d={d_u:+.3f}")
        csv_rows.append({"maqam":"micro_vs_ctrl","variant":vname,"key":"MW_U","value":round(float(U),4)})
        csv_rows.append({"maqam":"micro_vs_ctrl","variant":vname,"key":"MW_p","value":round(float(p_u),6)})
        csv_rows.append({"maqam":"micro_vs_ctrl","variant":vname,"key":"MW_d","value":round(float(d_u),4)})

    out = RESULTS / "respelling_test.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["maqam","variant","key","value"])
        w.writeheader(); w.writerows(csv_rows)
    print(f"\n  → Saved: {out}")

# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--skip-build", action="store_true", help="reuse existing variant databases")
    p.add_argument("--skip-run", action="store_true", help="reuse existing IDyOM .dat output")
    args = p.parse_args()

    if not args.skip_build: build_variant_databases()
    if not args.skip_run: run_idyom_on_variants()
    analyze()
