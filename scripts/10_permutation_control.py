"""Permutation control: redraws each note's comma label from its 12-TET class's empirical marginal (40 seeds), retrains IDyOM from scratch each time, and compares against the observed corpus."""

import argparse, csv, os, random, sqlite3, subprocess, sys, time
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from scipy import stats as ss

BASE       = Path("/Users/ugurozalp/makam_beklenti")          # local IDyOM output location; edit for your setup
OUTDIR     = BASE / "data" / "idyom_output"
SCRIPTS    = BASE / "scripts"
RESULTS    = Path(__file__).resolve().parent.parent / "results"
SQLITE_LIB = "/opt/homebrew/opt/sqlite/lib"                    # macOS IDyOM sqlite lib; edit for your setup

MAKAMS = ["ussak", "huseyni", "nihavent"]
LABELS = {"ussak": "Uşşak", "huseyni": "Hüseyni", "nihavent": "Nihâvend"}
MICRO  = ["ussak", "huseyni"]

ORIG_DATASET = {"ussak": 200, "huseyni": 202, "nihavent": 204}
ORIG_TET     = {"ussak": 201, "huseyni": 203, "nihavent": 205}
PERM_BASE    = {"ussak": 300, "huseyni": 340, "nihavent": 380}   # fresh, non-overlapping IDs
N_SEEDS      = 40

DH_BASELINE = {"ussak": 0.13778, "huseyni": 0.15268, "nihavent": 0.05010}  # from 08_baselines_and_diagnostics.py, section 1A

def koma_to_tet(k): return round(int(k) * 12 / 53)

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — build 40 permutation-variant databases per maqam (120 total)
# ═════════════════════════════════════════════════════════════════════════════

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
    print("STEP 1 — Building 40 permutation-variant databases per maqam (120 total) …")
    for makam in MAKAMS:
        cols, events = load_all_events(makam)
        compositions = load_compositions(makam)

        tet_koma_cnt = defaultdict(Counter)
        for ev in events:
            tet_koma_cnt[koma_to_tet(ev["CPITCH"])][ev["CPITCH"]] += 1
        marg = {}
        for tet, cnt in tet_koma_cnt.items():
            total = sum(cnt.values())
            komas = list(cnt.keys())
            weights = [cnt[k] / total for k in komas]
            marg[tet] = (komas, weights)

        base_id = PERM_BASE[makam]
        for seed in range(1, N_SEEDS + 1):
            rng = random.Random(1000 * base_id + seed)
            new_id = base_id + seed - 1
            new_events = []
            for ev in events:
                tet = koma_to_tet(ev["CPITCH"])
                komas, weights = marg[tet]
                new_cp = rng.choices(komas, weights=weights, k=1)[0]
                new_ev = dict(ev)
                new_ev["CPITCH"] = new_cp
                new_events.append(new_ev)
            write_variant_db(makam, new_id, f"{makam}_perm40_seed{seed}",
                             new_events, compositions, cols)
        print(f"  {makam}: datasets {base_id}-{base_id+N_SEEDS-1} written")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — run IDyOM on each of the 120 variant databases
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

def run_idyom_on_variants(only_ids=None):
    print("\nSTEP 2 — Running IDyOM on the 120 variant databases (this takes ~3.5 hours) …")
    jobs = [(makam, base + i, f"variant_{base + i}")
            for makam, base in PERM_BASE.items() for i in range(N_SEEDS)]
    if only_ids:
        want = set(only_ids)
        jobs = [j for j in jobs if j[1] in want]
    results = []
    for makam, dsid, subdir in jobs:
        ok, dt = run_one(makam, dsid, subdir)
        results.append((makam, dsid, subdir, ok, dt))
    n_ok = sum(1 for r in results if r[3])
    print(f"\n=== IDyOM runs: {n_ok}/{len(results)} succeeded ===")
    return results

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — analyze: compare permuted null distribution to the observed corpus
# ═════════════════════════════════════════════════════════════════════════════

def read_dat_full(path):
    with open(path) as f: hdr = f.readline().split()
    mid_i, nid_i = hdr.index("melody.id"), hdr.index("note.id")
    ic_i = hdr.index("cpitch.ic")
    out = {}
    with open(path) as f:
        f.readline()
        for line in f:
            pts = line.split()
            if pts[ic_i] == "NA": continue
            out[(int(pts[mid_i]), int(pts[nid_i]))] = float(pts[ic_i])
    return out

def get_dat(dir_path):
    files = sorted(dir_path.glob("*.dat"))
    if not files: raise FileNotFoundError(f"No .dat in {dir_path}")
    return files[0]

def cohens_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na-1)*np.var(a,ddof=1)+(nb-1)*np.var(b,ddof=1))/(na+nb-2))
    return (np.mean(a)-np.mean(b))/sp if sp > 0 else 0.0

def fmt_p(p): return "< .001" if p < .001 else f"= {p:.4f}"

def analyze():
    print("\nSTEP 3 — Analyzing: permuted null vs. observed (N=40 seeds/maqam) …")
    orig_tet  = {m: read_dat_full(get_dat(OUTDIR / m / "tet12")) for m in LABELS}
    orig_koma = {m: read_dat_full(get_dat(OUTDIR / m / "koma")) for m in LABELS}

    orig_dic_note, orig_dic_piece = {}, {}
    for m in LABELS:
        common = sorted(set(orig_koma[m]) & set(orig_tet[m]))
        dic = {k: orig_koma[m][k] - orig_tet[m][k] for k in common}
        orig_dic_note[m] = dic
        per_mid = defaultdict(list)
        for (mid, nid), v in dic.items(): per_mid[mid].append(v)
        orig_dic_piece[m] = {mid: np.mean(v) for mid, v in per_mid.items()}

    csv_rows = []
    struct_info_by_maqam = {}

    for m in LABELS:
        base = PERM_BASE[m]
        seed_note_means, seed_piece_arrays, missing = [], [], []
        for seed in range(N_SEEDS):
            dsid = base + seed
            dat_path = OUTDIR / m / f"variant_{dsid}"
            try:
                dv = read_dat_full(get_dat(dat_path))
            except FileNotFoundError:
                missing.append(dsid); continue
            common = sorted(set(dv) & set(orig_tet[m]))
            dic_perm = {k: dv[k] - orig_tet[m][k] for k in common}
            seed_note_means.append(np.mean(list(dic_perm.values())))
            per_mid = defaultdict(list)
            for (mid, nid), v in dic_perm.items(): per_mid[mid].append(v)
            seed_piece_arrays.append({mid: np.mean(v) for mid, v in per_mid.items()})

        n_ok = len(seed_note_means)
        if missing:
            print(f"  WARNING [{LABELS[m]}]: {len(missing)} seeds missing .dat output: {missing}")

        seed_note_means = np.array(seed_note_means)
        obs_note_mean = np.mean(list(orig_dic_note[m].values()))
        mean_perm  = float(seed_note_means.mean())
        sd_perm    = float(seed_note_means.std(ddof=1))
        p025, p975 = np.percentile(seed_note_means, [2.5, 97.5])
        pct_rank = 100 * np.mean(seed_note_means <= obs_note_mean)
        z = (obs_note_mean - mean_perm) / (sd_perm + 1e-12)
        emp_p = (np.sum(seed_note_means <= obs_note_mean) + 1) / (n_ok + 1)

        print(f"\n  {LABELS[m]}  (n_ok={n_ok}/{N_SEEDS} seeds)")
        print(f"    observed ΔIC (note-level, pooled)        = {obs_note_mean:+.5f}")
        print(f"    permuted null: mean={mean_perm:+.5f}  SD={sd_perm:.5f}")
        print(f"    permuted null: 2.5th pct={p025:+.5f}   97.5th pct={p975:+.5f}")
        print(f"    analytic ΔH_baseline (1A)                = {DH_BASELINE[m]:+.5f}")
        print(f"    observed value's percentile rank in null: {pct_rank:.2f}%   z = {z:+.3f}")
        print(f"    empirical p = (#perm <= obs + 1)/(N+1)   = {emp_p:.5f}")

        common_pieces = set(orig_dic_piece[m])
        for pm in seed_piece_arrays: common_pieces &= set(pm)
        common_pieces = sorted(common_pieces)
        perm_piece_stack = np.array([[pm[p] for p in common_pieces] for pm in seed_piece_arrays])
        mean_perm_piece = perm_piece_stack.mean(axis=0)
        obs_piece = np.array([orig_dic_piece[m][p] for p in common_pieces])
        struct_info = mean_perm_piece - obs_piece
        struct_info_by_maqam[m] = struct_info
        print(f"    piece-level structural info (perm-obs): mean={struct_info.mean():+.5f}  "
              f"n_pieces={len(common_pieces)}")

        for key, val in [
            ("n_seeds_ok", n_ok), ("obs_note_mean", obs_note_mean),
            ("perm_mean", mean_perm), ("perm_sd", sd_perm),
            ("perm_p2_5", p025), ("perm_p97_5", p975),
            ("dh_baseline_analytic", DH_BASELINE[m]),
            ("obs_percentile_rank_pct", pct_rank), ("z", z), ("emp_p", emp_p),
            ("struct_info_mean", struct_info.mean()), ("struct_info_med", np.median(struct_info)),
        ]:
            csv_rows.append({"maqam": LABELS[m], "key": key, "value": round(float(val), 6)})
        for i, v in enumerate(seed_note_means):
            csv_rows.append({"maqam": LABELS[m], "key": f"perm_note_mean_seed{i+1}", "value": round(float(v),6)})
        for i, v in enumerate(struct_info):
            csv_rows.append({"maqam": LABELS[m], "key": f"struct_info_piece_{i}", "value": round(float(v),6)})

    print("\nCross-maqam comparison on piece-level structural info")
    arrays = [struct_info_by_maqam[m] for m in MAKAMS]
    H, p_kw = ss.kruskal(*arrays)
    print(f"  Kruskal-Wallis: H={H:.4f}, p {fmt_p(p_kw)}")
    micro_si = np.concatenate([struct_info_by_maqam[m] for m in MICRO])
    ctrl_si  = struct_info_by_maqam["nihavent"]
    U, p_mw = ss.mannwhitneyu(micro_si, ctrl_si, alternative="two-sided")
    d_mw = cohens_d(micro_si, ctrl_si)
    print(f"  Mann-Whitney (micro vs Nihâvend): U={U:.1f}, p {fmt_p(p_mw)}, d={d_mw:+.3f}")
    for key, val in [("KW_H",H),("KW_p",p_kw),("MW_U",U),("MW_p",p_mw),("MW_d",d_mw)]:
        csv_rows.append({"maqam":"all","key":key,"value":round(float(val),6)})

    out = RESULTS / "permutation_control.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["maqam","key","value"])
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
