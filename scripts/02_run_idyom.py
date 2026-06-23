# Generate and run IDyOM Lisp scripts for all six dataset/representation combinations.

import subprocess
from pathlib import Path

# ── Configure these paths before running ──────────────────────────────────────
OUTPUT_DIR  = Path("data/idyom_output")
IDYOM_ROOT  = Path("/path/to/idyom")   # directory containing idyom.asd
SBCL        = "sbcl"                   # path to SBCL binary if not on PATH

DATASET_IDS = {
    "ussak":    {"koma": 200, "tet12": 201},
    "huseyni":  {"koma": 202, "tet12": 203},
    "nihavent": {"koma": 204, "tet12": 205},
}

CV_FOLDS = 10

LISP_TEMPLATE = """\
(defvar *idyom-root* "{idyom_root}")
(ql:quickload "clsql-sqlite3" :silent t)
(ql:quickload "idyom" :silent t)
(clsql:connect (list "{db_path}") :database-type :sqlite3 :if-exists :old)
(idyom:idyom {dataset_id}
             '(cpitch)
             '(cpint cpitch)
             :models :both+
             :k {cv_folds}
             :detail 3
             :output-path "{output_dir}/"
             :overwrite t)
(quit)
"""

def run_idyom(makam, arm, dataset_id):
    db_path    = OUTPUT_DIR / makam / arm / "idyom.db"
    out_dir    = OUTPUT_DIR / makam / arm
    lisp_src   = LISP_TEMPLATE.format(
        idyom_root = IDYOM_ROOT,
        db_path    = db_path,
        dataset_id = dataset_id,
        cv_folds   = CV_FOLDS,
        output_dir = out_dir,
    )
    lisp_file = out_dir / "run.lisp"
    lisp_file.write_text(lisp_src)
    print(f"Running IDyOM: {makam}/{arm} (dataset {dataset_id}) …")
    result = subprocess.run(
        [SBCL, "--load", str(lisp_file)],
        capture_output=True, text=True,
        env={"DYLD_LIBRARY_PATH": "/opt/homebrew/opt/sqlite/lib"},
    )
    if result.returncode != 0:
        print(f"  ERROR:\n{result.stderr[-2000:]}")
    else:
        print(f"  Done.")

def main():
    for makam in ["ussak", "huseyni", "nihavent"]:
        for arm, did in DATASET_IDS[makam].items():
            run_idyom(makam, arm, did)

if __name__ == "__main__":
    main()
