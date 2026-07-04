# Compute Schellenberg (1996) 5-factor implication-realisation scores for koma and 12-TET pitch sequences.

import sqlite3
from pathlib import Path
from collections import defaultdict

import numpy as np

# ── Configure these paths before running ──────────────────────────────────────
SYMBTR_TXT_DIR = Path("/path/to/SymbTr/txt")
OUTPUT_DIR     = Path("data/idyom_output")

DATASET_IDS = {
    "ussak":    {"koma": 200, "tet12": 201},
    "huseyni":  {"koma": 202, "tet12": 203},
    "nihavent": {"koma": 204, "tet12": 205},
}

# Proportionally scaled thresholds (1 semitone ≈ 53/12 Holder commas)
KOMA_LARGE, KOMA_SMALL = 27, 9   # ≈ 6 ST, ≈ 2 ST
TET_LARGE,  TET_SMALL  = 6,  2   # semitones

# ── IR model ──────────────────────────────────────────────────────────────────

def ir_score(I1, I2, large, small):
    """
    5-factor IR score for a pitch triple given intervals I1 and I2.
    Returns a float in [0, 5]; higher = more expected.
    Factors: RDIR, DIFF, RRET, PROX, CLOS (Schellenberg 1996).
    """
    a1, a2   = abs(I1), abs(I2)
    same_dir = (I1 > 0 and I2 > 0) or (I1 < 0 and I2 < 0)

    rdir = (1.0 if same_dir else 0.0) if a1 > large else \
           (0.0 if same_dir else 1.0) if a1 <= small else 0.0
    diff = 1.0 if a2 < a1 else 0.0
    rret = 1.0 if abs(I1 + I2) <= small else 0.0
    prox = max(0.0, 1.0 - a2 / large) if large > 0 else 0.0
    clos = 1.0 if (not same_dir) and (a2 <= small) else 0.0

    return rdir + diff + rret + prox + clos

# ── Data helpers ──────────────────────────────────────────────────────────────

def _int(s):
    try: return int(s)
    except (ValueError, TypeError): return None

def parse_txt(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            pts = line.rstrip("\n").split("\t")
            if len(pts) < 9: continue
            if _int(pts[1]) != 9: continue
            koma = _int(pts[4])
            if not koma or koma <= 0: continue
            rows.append((koma, round(koma * 12 / 53)))
    return rows

def get_pieces(makam):
    db = OUTPUT_DIR / makam / "koma" / "idyom.db"
    with sqlite3.connect(db) as c:
        return c.execute(
            "SELECT COMPOSITION_ID, DESCRIPTION FROM mtp_composition "
            "WHERE DATASET_ID=? ORDER BY COMPOSITION_ID",
            (DATASET_IDS[makam]["koma"],)
        ).fetchall()

def get_karar(makam):
    db = OUTPUT_DIR / makam / "koma" / "idyom.db"
    with sqlite3.connect(db) as c:
        rows = c.execute(
            "SELECT COMPOSITION_ID, CPITCH FROM mtp_event "
            "WHERE DATASET_ID=? ORDER BY COMPOSITION_ID, ONSET",
            (DATASET_IDS[makam]["koma"],)
        ).fetchall()
    last = {}
    for cid, cp in rows: last[cid] = cp
    return last

# ── Main computation ──────────────────────────────────────────────────────────

def compute_ir(makam):
    """
    Returns:
      piece_records — list of {name, ir_koma, ir_tet, delta_ir, n_triples}
      deg_records   — {degree: [delta_ir]} for degree-level analysis
    """
    pieces    = get_pieces(makam)
    karar_map = get_karar(makam)
    piece_records = []
    deg_records   = defaultdict(list)

    for cid_db, name in pieces:
        txt = SYMBTR_TXT_DIR / (name + ".txt")
        if not txt.exists(): continue
        notes = parse_txt(txt)
        if len(notes) < 3: continue
        karar = karar_map.get(cid_db)
        if karar is None: continue

        komas  = [n[0] for n in notes]
        tet12s = [n[1] for n in notes]
        sks, sts = [], []

        for i in range(len(notes) - 2):
            sk = ir_score(komas[i+1]-komas[i],   komas[i+2]-komas[i+1],   KOMA_LARGE, KOMA_SMALL)
            st = ir_score(tet12s[i+1]-tet12s[i], tet12s[i+2]-tet12s[i+1], TET_LARGE,  TET_SMALL)
            sks.append(sk); sts.append(st)
            deg_records[(komas[i+2] - karar) % 53].append(sk - st)

        if not sks: continue
        piece_records.append({
            "name":     name,
            "ir_koma":  np.mean(sks),
            "ir_tet":   np.mean(sts),
            "delta_ir": np.mean(sks) - np.mean(sts),
            "n":        len(sks),
        })

    return piece_records, deg_records

def main():
    for makam in ["ussak", "huseyni", "nihavent"]:
        recs, _ = compute_ir(makam)
        dirs = [r["delta_ir"] for r in recs]
        print(f"{makam}: {len(recs)} pieces  "
              f"IR_koma={np.mean([r['ir_koma'] for r in recs]):.4f}  "
              f"IR_tet={np.mean([r['ir_tet'] for r in recs]):.4f}  "
              f"ΔIR mean={np.mean(dirs):+.5f}")

if __name__ == "__main__":
    main()
