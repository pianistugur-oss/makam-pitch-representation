"""Publication figure (Figure 2): piece-level ΔIC violin plot, comma-level vs. 12-TET information content, per makam."""

import sqlite3
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE       = Path("/Users/ugurozalp/makam_beklenti")   # local IDyOM output location; edit for your setup
SYMBTR_DIR = Path("/Users/ugurozalp/Downloads/SymbTr-master/txt")   # local SymbTr checkout; edit for your setup
OUTDIR     = BASE / "data" / "idyom_output"
FIGDIR     = Path(__file__).resolve().parent.parent.parent / "figures"

MAKAMS  = ["ussak", "huseyni", "nihavent"]
LABELS  = {"ussak": "Uşşak", "huseyni": "Hüseyni", "nihavent": "Nihâvend"}
COLOR   = {"ussak": "#C94B2B", "huseyni": "#C49A00", "nihavent": "#2B6CB0"}
DATASET = {"ussak": {"koma": 200, "tet12": 201},
           "huseyni": {"koma": 202, "tet12": 203},
           "nihavent": {"koma": 204, "tet12": 205}}

plt.rcParams.update({"font.size": 10, "axes.titlesize": 10,
                      "axes.labelsize": 10, "xtick.labelsize": 9,
                      "ytick.labelsize": 9})

# ── shared helpers (same as figures_english.py) ────────────────────────────────

def read_dat(p):
    rows = []
    with open(p) as f:
        hdr = f.readline().split()
        for line in f:
            pts = line.split(); row = dict(zip(hdr, pts))
            if row.get("cpitch.ic", "NA") == "NA": continue
            rows.append({"mid": int(row["melody.id"]), "nid": int(row["note.id"]),
                         "cp": int(row["cpitch"]), "ic": float(row["cpitch.ic"]),
                         "name": row.get("melody.name", "").strip('"')})
    return rows

def get_karar(makam):
    db = OUTDIR / makam / "koma" / "idyom.db"
    with sqlite3.connect(db) as c:
        rows = c.execute("SELECT COMPOSITION_ID,CPITCH FROM mtp_event "
                         "WHERE DATASET_ID=? ORDER BY COMPOSITION_ID,ONSET",
                         (DATASET[makam]["koma"],)).fetchall()
    last = {}
    for cid, cp in rows: last[cid] = cp
    return last

def build_records(makam):
    dat_k = sorted((OUTDIR/makam/"koma").glob("*.dat"))[0]
    dat_t = sorted((OUTDIR/makam/"tet12").glob("*.dat"))[0]
    karar = get_karar(makam)
    rk    = read_dat(dat_k); rt = read_dat(dat_t)
    idx_t = {(r["mid"], r["nid"]): r["ic"] for r in rt}
    names = {r["mid"]: r["name"] for r in rk}
    recs  = []
    for r in rk:
        ic_t = idx_t.get((r["mid"], r["nid"]))
        if ic_t is None: continue
        k = karar.get(r["mid"]-1)   # DB 0-based
        if k is None: continue
        recs.append({"mid": r["mid"], "name": names[r["mid"]],
                     "cp_koma": r["cp"], "cp_tet": round(r["cp"]*12/53),
                     "ic_koma": r["ic"], "ic_tet": ic_t,
                     "delta_ic": r["ic"]-ic_t,
                     "deg": (r["cp"]-k) % 53})
    return recs

def piece_dic(recs):
    """Per-piece mean ΔIC: {(mid, name): value}"""
    per = defaultdict(list); names = {}
    for r in recs:
        per[r["mid"]].append(r["delta_ic"]); names[r["mid"]] = r["name"]
    return {(m, names[m]): np.mean(v) for m, v in per.items()}

def bootstrap_ci(data, n=5000, alpha=0.05, seed=42):
    rng  = np.random.default_rng(seed)
    data = np.asarray(data)
    boot = [np.mean(rng.choice(data, len(data), replace=True)) for _ in range(n)]
    return np.percentile(boot, 100*alpha/2), np.percentile(boot, 100*(1-alpha/2))

# ── figure ─────────────────────────────────────────────────────────────────────

def fig_pub_violin():
    all_recs   = {m: build_records(m) for m in MAKAMS}
    piece_data = {m: list(piece_dic(all_recs[m]).values()) for m in MAKAMS}

    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    data = [piece_data[m] for m in MAKAMS]
    vp   = ax.violinplot(data, positions=[1,2,3], showmedians=True, showextrema=False, widths=0.55)
    for body, m in zip(vp["bodies"], MAKAMS):
        body.set_facecolor(COLOR[m]); body.set_alpha(0.72); body.set_edgecolor("none")
    vp["cmedians"].set_color("white"); vp["cmedians"].set_linewidth(2)

    rng = np.random.default_rng(42)
    for pos, m in zip([1,2,3], MAKAMS):
        jx = pos + rng.uniform(-.09, .09, len(piece_data[m]))
        ax.scatter(jx, piece_data[m], color=COLOR[m], s=11, alpha=0.45, lw=0, zorder=3)

    for pos, m in zip([1,2,3], MAKAMS):
        mn = np.mean(piece_data[m])
        lo, hi = bootstrap_ci(piece_data[m])
        ax.text(pos, ax.get_ylim()[1]*.96 if ax.get_ylim()[1]>0 else -.03,
                f"M = {mn:+.3f}", ha="center", fontsize=8, color=COLOR[m],
                fontweight="bold")

    ax.axhline(0, color="#888", lw=0.8, ls="--")
    ax.set_xticks([1,2,3])
    ax.set_xticklabels([LABELS[m] for m in MAKAMS], fontsize=11)
    ax.set_ylabel("Mean ΔIC per piece  [bits]")
    ax.set_title("Comma-Level vs. 12-TET Information Content\n(piece level)", fontsize=10)
    ax.spines[["top","right"]].set_visible(False)

    fig.tight_layout()
    FIGDIR.mkdir(exist_ok=True)
    for ext in ("png","pdf"):
        out = FIGDIR / f"pub_violin_piece_dic_eng.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {FIGDIR / 'pub_violin_piece_dic_eng.png'} (+ .pdf)")

if __name__ == "__main__":
    fig_pub_violin()
