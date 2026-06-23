# Parse SymbTr TXT files and insert koma-53 and 12-TET pitch sequences into IDyOM-compatible SQLite databases.

import sqlite3, statistics
from pathlib import Path

# ── Configure these paths before running ──────────────────────────────────────
SYMBTR_TXT_DIR = Path("/path/to/SymbTr/txt")
OUTPUT_DIR     = Path("data/idyom_output")

TIMEBASE    = 96
WHOLE_TICKS = 4 * TIMEBASE   # 384 ticks per whole note

KARAR_MOD53 = {"ussak": 40, "huseyni": 40, "nihavent": 31}

DATASET_IDS = {
    "ussak":    {"koma": 200, "tet12": 201},
    "huseyni":  {"koma": 202, "tet12": 203},
    "nihavent": {"koma": 204, "tet12": 205},
}

MIDC = {"koma": 265, "tet12": 60}

# ── SymbTr TXT parser ─────────────────────────────────────────────────────────
# Columns (tab-separated): Sira Kod Nota53 NotaAE Koma53 KomaAE Pay Payda Ms LNS Bas Soz1 Offset

def _int(s):
    try: return int(s)
    except (ValueError, TypeError): return None

def parse_txt(path):
    rows = []; onset = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9: continue
            if _int(parts[1]) != 9: continue
            koma53 = _int(parts[4]) if parts[4].strip() else -1
            if koma53 is None: koma53 = -1
            pay   = _int(parts[6]) or 0
            payda = _int(parts[7]) or 0
            dur   = max(1, round(pay * WHOLE_TICKS / payda)) if payda > 0 else 1
            rows.append((koma53, onset, dur))
            onset += dur
    return rows

def find_karar(rows):
    for koma53, _, _ in reversed(rows):
        if koma53 > 0: return koma53
    return None

def is_outlier(rows, makam):
    karar = find_karar(rows)
    return karar is None or (karar % 53) != KARAR_MOD53[makam]

# ── IDyOM SQLite schema ───────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS mtp_dataset (
    DATASET_ID INTEGER PRIMARY KEY, DESCRIPTION VARCHAR(255),
    TIMEBASE INTEGER, MIDC INTEGER);
CREATE TABLE IF NOT EXISTS mtp_composition (
    COMPOSITION_ID INTEGER, DATASET_ID INTEGER,
    TIMEBASE INTEGER, DESCRIPTION VARCHAR(255),
    PRIMARY KEY (DATASET_ID, COMPOSITION_ID));
CREATE TABLE IF NOT EXISTS mtp_event (
    EVENT_ID INTEGER PRIMARY KEY, COMPOSITION_ID INTEGER, DATASET_ID INTEGER,
    ONSET INTEGER, CPITCH INTEGER, MPITCH INTEGER, ACCIDENTAL INTEGER,
    DUR INTEGER, DELTAST INTEGER, BIOI INTEGER, KEYSIG INTEGER, MODE INTEGER,
    BARLENGTH INTEGER, PULSES INTEGER, PHRASE INTEGER, TEMPO INTEGER,
    DYN INTEGER, ORNAMENT INTEGER, COMMA INTEGER, ARTICULATION INTEGER,
    VERTINT12 INTEGER, VOICE INTEGER);
"""

class DbWriter:
    def __init__(self, db_path, dataset_id, description, midc):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.dataset_id = dataset_id
        self.comp_id = 0; self.event_id = 1; self.note_count = 0
        self.conn.executescript(SCHEMA_SQL)
        self.conn.execute("INSERT OR REPLACE INTO mtp_dataset VALUES (?,?,?,?)",
                          (dataset_id, description, TIMEBASE, midc))

    def add(self, name, notes):
        self.conn.execute("INSERT OR REPLACE INTO mtp_composition VALUES (?,?,?,?)",
                          (self.comp_id, self.dataset_id, TIMEBASE, name))
        rows = []
        prev = 0
        for i, (cp, onset, dur) in enumerate(notes):
            rows.append((self.event_id+i, self.comp_id, self.dataset_id, onset, cp,
                         0, 0, dur, 0, onset-prev if i>0 else 0,
                         0, 9, 384, 4, 0, 60, 64, 0, 0, 0, 0, 0))
            prev = onset
        self.conn.executemany("INSERT INTO mtp_event VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        self.event_id += len(notes); self.comp_id += 1; self.note_count += len(notes)

    def close(self):
        self.conn.commit(); self.conn.close()

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    for makam in ["ussak", "huseyni", "nihavent"]:
        txt_files = sorted(SYMBTR_TXT_DIR.glob(f"{makam}--*.txt"))
        w_k = DbWriter(OUTPUT_DIR/makam/"koma"/"idyom.db",
                       DATASET_IDS[makam]["koma"], f"SymbTr {makam} koma53", MIDC["koma"])
        w_t = DbWriter(OUTPUT_DIR/makam/"tet12"/"idyom.db",
                       DATASET_IDS[makam]["tet12"], f"SymbTr {makam} 12tet", MIDC["tet12"])
        n_ok = n_out = n_emp = 0
        for fpath in txt_files:
            rows = parse_txt(fpath)
            if not rows: n_emp += 1; continue
            if is_outlier(rows, makam): n_out += 1; continue
            valid = [(k, o, d) for k,o,d in rows if k > 0]
            if len(valid) < 3: n_emp += 1; continue
            w_k.add(fpath.stem, [(k,o,d) for k,o,d in valid])
            w_t.add(fpath.stem, [(round(k*12/53),o,d) for k,o,d in valid])
            n_ok += 1
        w_k.close(); w_t.close()
        print(f"{makam}: {n_ok} pieces, {w_k.note_count} notes  ({n_out} outliers removed, {n_emp} skipped)")

if __name__ == "__main__":
    main()
