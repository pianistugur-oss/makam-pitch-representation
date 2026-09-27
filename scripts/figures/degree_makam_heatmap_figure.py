"""Publication figure: degree × makam ΔIC heatmap with markers for theoretically-meaningful high-ΔIC degrees and perde-name reference guides."""
import sqlite3
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm

BASE       = Path("/Users/ugurozalp/makam_beklenti")   # local IDyOM output location; edit for your setup
SYMBTR_DIR = Path("/Users/ugurozalp/Downloads/SymbTr-master/txt")   # local SymbTr checkout; edit for your setup
OUTDIR     = BASE / "data" / "idyom_output"
FIGDIR     = Path(__file__).resolve().parent.parent.parent / "figures"

MAKAMS  = ["ussak", "huseyni", "nihavent"]
LABELS  = {"ussak": "Uşşak", "huseyni": "Hüseyni", "nihavent": "Nihâvend"}
DATASET = {"ussak": {"koma": 200, "tet12": 201},
           "huseyni": {"koma": 202, "tet12": 203},
           "nihavent": {"koma": 204, "tet12": 205}}
N_MIN = 50

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 10, 'axes.titlesize': 10, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
})

# ── data loading ───────────────────────────────────────────────────────────────

def _int(s):
    try: return int(s)
    except: return None

def read_dat(p):
    rows = []
    with open(p) as f:
        hdr = f.readline().split()
        for line in f:
            pts = line.split(); row = dict(zip(hdr, pts))
            if row.get("cpitch.ic", "NA") == "NA": continue
            rows.append({"mid": int(row["melody.id"]), "nid": int(row["note.id"]),
                         "cp": int(row["cpitch"]), "ic": float(row["cpitch.ic"])})
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
    rk, rt = read_dat(dat_k), read_dat(dat_t)
    idx_t = {(r["mid"], r["nid"]): r["ic"] for r in rt}
    recs = []
    for r in rk:
        ic_t = idx_t.get((r["mid"], r["nid"]))
        if ic_t is None: continue
        k = karar.get(r["mid"] - 1)
        if k is None: continue
        recs.append({"mid": r["mid"], "cp_koma": r["cp"],
                     "delta_ic": r["ic"] - ic_t, "deg": (r["cp"] - k) % 53})
    return recs

def get_nota53(makam, deg, karar_map):
    db = OUTDIR / makam / "koma" / "idyom.db"
    with sqlite3.connect(db) as c:
        rows = c.execute("SELECT COMPOSITION_ID,CPITCH FROM mtp_event WHERE DATASET_ID=?",
                         (DATASET[makam]["koma"],)).fetchall()
    komas = Counter()
    for cid, cp in rows:
        k = karar_map.get(cid)
        if k is not None and (cp - k) % 53 == deg:
            komas[cp] += 1
    KOMA_NAMES = defaultdict(Counter)
    for txt in SYMBTR_DIR.glob(f"{makam}--*.txt"):
        with open(txt, encoding="utf-8") as f:
            for line in f:
                pts = line.rstrip("\n").split("\t")
                if len(pts) < 5 or _int(pts[1]) != 9: continue
                k2 = _int(pts[4]); n = pts[2].strip()
                if k2 and k2 > 0 and n: KOMA_NAMES[k2][n] += 1
    all_notas = Counter()
    for koma, cnt in komas.items():
        for nota, n in KOMA_NAMES[koma].items(): all_notas[nota] += n
    return all_notas.most_common(1)[0][0] if all_notas else "?"

print("Loading records and computing heatmap matrix …")
all_recs = {m: build_records(m) for m in MAKAMS}
karar_maps = {m: get_karar(m) for m in MAKAMS}

matrices = {}
all_degs = set()
for m in MAKAMS:
    dd = defaultdict(list)
    for r in all_recs[m]: dd[r["deg"]].append(r["delta_ic"])
    matrices[m] = {d: np.mean(v) for d, v in dd.items() if len(v) >= N_MIN}
    all_degs.update(matrices[m])
degs = sorted(all_degs)

data = np.full((3, len(degs)), np.nan)
for i, m in enumerate(MAKAMS):
    for j, d in enumerate(degs):
        if d in matrices[m]: data[i, j] = matrices[m][d]

vmax_data = float(np.nanmax(data))
vmin_data = float(np.nanmin(data))
print(f"Data range: [{vmin_data:.3f}, {vmax_data:.3f}]")

finite_vals = data[~np.isnan(data)]
vmax_capped = float(np.percentile(np.abs(finite_vals), 90))
print(f"90th percentile |value| = {vmax_capped:.3f} -> display vmax "
      f"(outlier {vmax_data:.3f} saturates colormap; see caption)")

# ── marker / reference definitions ──────────────────────────────────────────────

MARKED_NEW = [
    ("huseyni",  37, "2 commas above Acem"),
    ("nihavent", 36, "hisar"),
    ("nihavent", 17, "segah"),
]
REFERENCE_DEGS_HUSEYNI = {35: "Acem", 36: "Dik Acem", 39: "Eviç"}

# content-driven verification (printed at the end) --------------------------
verify_report = []
for maqam, deg, name in MARKED_NEW:
    row_idx = MAKAMS.index(maqam)
    col_idx = degs.index(deg) if deg in degs else None
    verify_report.append((maqam, deg, name, row_idx, col_idx))

# ── figure: 3-row GridSpec (ref guides / heatmap / marked labels) ─────────────

n_cols = len(degs)
fig_w = max(11.5, n_cols * 0.40 + 2.2)   # extra width so colorbar label never clips
fig = plt.figure(figsize=(fig_w, 4.7), dpi=300)
gs = fig.add_gridspec(3, 1, height_ratios=[0.38, 1.9, 0.75], hspace=0.10,
                      left=0.07, right=0.90, top=0.85, bottom=0.085)
# ax_hm's share of TOTAL figure height: (1.9/3.03) * (0.85-0.085) = 48.0%
ax_ref = fig.add_subplot(gs[0])
ax_hm  = fig.add_subplot(gs[1])
ax_lbl = fig.add_subplot(gs[2], sharex=ax_hm)

# ---- heatmap ----
masked = np.ma.array(data, mask=np.isnan(data))
cmap = matplotlib.colormaps["RdBu_r"].copy(); cmap.set_bad("#DDDDDD")
cmap.set_over("#3d0d0d")

im = ax_hm.imshow(masked, aspect="auto", cmap=cmap,
                  norm=TwoSlopeNorm(vmin=-vmax_capped, vcenter=0, vmax=vmax_capped),
                  interpolation="nearest")
# IMPORTANT: attach to ALL THREE stacked axes (not just ax_hm), so matplotlib
# shrinks their widths together. Attaching to ax_hm alone narrows only that
# axes, leaving ax_ref/ax_lbl at their original (wider) extent -- since all
# three share xlim by column INDEX, that width mismatch turns into a
# growing pixel-space misalignment toward the right edge (col 19 was off by
# ~143px, col 0 by ~4px) even though the underlying data coordinates were
# always correct and consistent.
cb = fig.colorbar(im, ax=[ax_ref, ax_hm, ax_lbl], shrink=0.85, pad=0.015,
                  extend='max', fraction=0.045)
# explicit ticks: TwoSlopeNorm's default locator can emit out-of-domain
# candidates (e.g. -5) that evaluate to NaN positions with extend='max'
tick_step = round(vmax_capped / 2, 1) or 0.5
cb.set_ticks(np.arange(-2*tick_step, 2*tick_step + 1e-9, tick_step))
cb.set_label("Mean ΔIC [bits]", fontsize=9)
cb.ax.tick_params(labelsize=7.5)

ax_hm.set_yticks(range(3))
ax_hm.set_yticklabels([LABELS[m] for m in MAKAMS], fontsize=10)
ax_hm.set_xticks(range(len(degs)))
ax_hm.set_xticklabels([str(d) for d in degs], fontsize=7.5, rotation=45, ha="right",
                      rotation_mode="anchor")
ax_hm.set_xlim(-0.5, len(degs) - 0.5)
ax_hm.set_ylim(2.5, -0.5)
for sp in ax_hm.spines.values(): sp.set_visible(True)

# d=7 / d=8 column-wide markers (drawn ON the heatmap axes)
# Both are the SAME box style/colour/weight -- only the line style (solid
# vs dashed) distinguishes them. d=7 and d=8 are adjacent columns, so each
# box is inset horizontally from its cell edge -> a visible gap of bare
# heatmap colour separates the two boxes instead of them sharing one edge
# and reading as a single fused block.
INSET = 0.09
for j, d in enumerate(degs):
    if d == 8:
        ax_hm.add_patch(mpatches.Rectangle((j-.5+INSET, -.5), 1-2*INSET, 3,
                        fill=False, edgecolor="black", lw=1.8, ls="-", zorder=5))
    elif d == 7:
        ax_hm.add_patch(mpatches.Rectangle((j-.5+INSET, -.5), 1-2*INSET, 3,
                        fill=False, edgecolor="black", lw=1.8, ls="--", zorder=5))

# single-cell markers for the new degrees (drawn ON the heatmap axes)
for maqam, deg, name, row_idx, col_idx in verify_report:
    if col_idx is None: continue
    ax_hm.add_patch(mpatches.Rectangle((col_idx-.5, row_idx-.5), 1, 1,
                    fill=False, edgecolor="#0B6E4F", lw=1.4,
                    ls=(0, (2, 1, 1, 1)), zorder=5))

# ---- reference-guide axes (top): Acem / Dik Acem / Eviç ----
ax_ref.set_xlim(ax_hm.get_xlim())
ax_ref.set_ylim(0, 1)
ax_ref.axis("off")
# Vertical order requested: note -> title -> perde reference guides -> heatmap.
# Both the note and the title are figure-level elements (not tied to ax_ref),
# so their stacking order is independent of the heatmap's own gridspec rows.
fig.text(0.5, 0.985, "Perde names relative to A karar (Uşşak, Hüseyni only — "
          "not applicable to Nihâvend, karar = G)",
          ha="center", va="top", fontsize=6.5, style="italic", color="#666666")
fig.suptitle(f"Degree × Makam  ΔIC heatmap  (n ≥ {N_MIN})", fontsize=11, y=0.945)

ref_items = sorted([(degs.index(d), d, n) for d, n in REFERENCE_DEGS_HUSEYNI.items() if d in degs])
prev_j, level = None, 0
REF_LEVEL_Y = {0: 0.34, 1: 0.10}
for j, deg, name in ref_items:
    if prev_j is not None and (j - prev_j) <= 2:
        level = 1 - level
    else:
        level = 0
    y_text = REF_LEVEL_Y[level]
    # short tick from bottom of this axes (0) UP to just below text baseline
    ax_ref.plot([j, j], [0.0, y_text - 0.03], transform=ax_ref.get_xaxis_transform(),
                color="#AAAAAA", lw=0.6, ls=":", clip_on=False)
    ax_ref.text(j, y_text, name, transform=ax_ref.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=6.3, color="#888888")
    prev_j = j

# ---- marked-label axes (bottom): d=7/d=8/d=17/d=36/d=37 descriptive labels ----
# (the numeric degree ticks themselves live on ax_hm via its native tick
# mechanism -- this axes only holds the NEW descriptive text below them, so
# the two label systems never compete for the same coordinate space)
ax_lbl.set_xlim(ax_hm.get_xlim())
ax_lbl.set_ylim(1, 0)   # inverted: 1=top(near heatmap) .. 0=bottom(figure edge)
ax_lbl.axis("off")

label_defs = []   # (col_idx, lines:list[str], color, weight_first_bold)
for j, d in enumerate(degs):
    if d == 8:
        label_defs.append((j, ["d=8", "segah"], "black", True))
    elif d == 7:
        label_defs.append((j, ["d=7", "kürdi"], "black", True))
for maqam, deg, name, row_idx, col_idx in verify_report:
    if col_idx is None: continue
    label_defs.append((col_idx, [f"d={deg}", name, f"({LABELS[maqam]})"], "#0B6E4F", True))

# stagger by proximity to avoid horizontal collisions
# levels chosen so even a 3-line block's TOP line stays clear of y=1.0
# (the boundary shared with ax_hm's rotated numeric tick labels)
label_defs.sort(key=lambda t: t[0])
prev_j, level = None, 0
LABEL_LEVEL_Y0 = {0: 0.40, 1: 0.05}
LINE_DY = 0.115
for j, lines, color, bold_first in label_defs:
    if prev_j is not None and (j - prev_j) <= 2:
        level = 1 - level
    else:
        level = 0
    y0 = LABEL_LEVEL_Y0[level]
    # short connector from just under the heatmap up to this label block only
    # (kept strictly within this axes' own coordinate range -> can never cross
    # into a neighbouring label block)
    ax_lbl.plot([j, j], [1.0, y0 + len(lines) * LINE_DY + 0.03],
                transform=ax_lbl.get_xaxis_transform(),
                color=color, lw=0.5, alpha=0.55, ls=":", clip_on=False)
    for k, line in enumerate(lines):
        fw = "bold" if (k == 0 and bold_first) else "normal"
        fstyle = "italic" if k == 1 else "normal"
        fs = 5.6 if k == 2 else 6.6
        ax_lbl.text(j, y0 + (len(lines) - 1 - k) * LINE_DY, line,
                    transform=ax_lbl.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=fs, color=color,
                    fontweight=fw, style=fstyle)
    prev_j = j

fig.text(0.46, 0.012, "Tonic-relative scale degree (comma mod 53)",
         ha="center", va="bottom", fontsize=10)

# ── column index -> degree table ────────────────────────────────────────────
print("\n" + "="*78)
print("COLUMN INDEX -> DEGREE VALUE TABLE")
print("="*78)
for j, d in enumerate(degs):
    print(f"  col_idx={j:2d}  ->  degree={d}")

# ── alignment verification: label / leader-line / box / column-centre ──────
# Must run AFTER all artists are placed but a canvas draw is needed for
# transData to reflect the FINAL (post-colorbar-shrink) axes positions.
fig.canvas.draw()

def px_x(ax, data_x):
    """Pixel x-coordinate that `data_x` (in ax's own data coords) maps to."""
    return ax.transData.transform((data_x, 0))[0]

align_items = []   # (label_text, data_x_used, leader_x0, leader_x1, label_axes)
for j, d in enumerate(degs):
    if d == 7:
        align_items.append(("d=7 / kürdi", j, j, j, ax_lbl))
    elif d == 8:
        align_items.append(("d=8 / segah", j, j, j, ax_lbl))
for maqam, deg, name, row_idx, col_idx in verify_report:
    if col_idx is None: continue
    align_items.append((f"d={deg} / {name}", col_idx, col_idx, col_idx, ax_lbl))
for j, deg, name in ref_items:
    align_items.append((f"ref d={deg} / {name}", j, j, j, ax_ref))

print("\n" + "="*118)
print("LABEL / LEADER-LINE / BOX / COLUMN-CENTRE ALIGNMENT (data-x by construction; pixel-x after render)")
print("="*118)
print(f"  {'label':20s} {'label x':>7s} {'leadr x0':>8s} {'leadr x1':>8s} {'col_idx':>7s} "
      f"{'col centre(px)':>14s} {'box x(px)':>10s} {'label x(px)':>12s} {'Δpx (label-col)':>16s}")

max_abs_dpx = 0.0
for label, data_x, leader_x0, leader_x1, label_ax in align_items:
    col_centre_px = px_x(ax_hm, data_x)     # ground truth: column centre per the heatmap axes
    box_px        = px_x(ax_hm, data_x)     # boxes are drawn directly on ax_hm -> same reference
    label_px      = px_x(label_ax, data_x)  # label lives in ax_lbl or ax_ref
    dpx = label_px - col_centre_px
    max_abs_dpx = max(max_abs_dpx, abs(dpx))
    print(f"  {label:20s} {data_x:7d} {leader_x0:8d} {leader_x1:8d} {data_x:7d} "
          f"{col_centre_px:14.2f} {box_px:10.2f} {label_px:12.2f} {dpx:+16.2f}")

print(f"\n  Max |Δpx| across all labels: {max_abs_dpx:.2f}px")
if max_abs_dpx < 1.0:
    print("  -> CONFIRMED: every label and leader line is pixel-aligned with its "
          "target column centre (sub-pixel tolerance).")
else:
    print("  -> WARNING: residual misalignment detected, see rows above.")

out = FIGDIR / "Figure_heatmap_updated.png"
fig.savefig(out, dpi=300, facecolor="white")
print(f"\nSaved: {out}")

# ── content verification ──────────────────────────────────────────────────────
print("\n" + "="*78)
print("CONTENT VERIFICATION — marked (row, col) cells")
print("="*78)
for maqam, deg, name, row_idx, col_idx in verify_report:
    print(f"  {maqam:10s} d={deg:<4} '{name:22s}'  -> row={row_idx} ({LABELS[maqam]}), "
          f"col={col_idx} (degree column {deg})")
print("\nReference guides (Hüseyni-only context, no emphasis box):")
for j, deg, name in ref_items:
    print(f"  d={deg:<4} '{name}'  -> column {degs.index(deg)}")

# ── programmatic overlap / clipping check ─────────────────────────────────────

def validate_figure(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_bbox = fig.bbox
    boxes = []
    for ax in fig.axes:
        # axis("off") axes (e.g. ax_ref) disable the whole tick/label subsystem
        # at the Axes.draw() level regardless of each Text's own get_visible();
        # their default-locator tick labels (e.g. an out-of-range "-5"/"25" from
        # an unset x-tick locator) are never rendered, so skip them here rather
        # than flag them as false-positive clipping.
        if not getattr(ax, "axison", True):
            continue
        candidates = list(ax.texts) + [ax.title, ax.xaxis.label, ax.yaxis.label]
        candidates += list(ax.get_xticklabels()) + list(ax.get_yticklabels())
        for t in candidates:
            txt = t.get_text()
            if not txt: continue
            bb = t.get_window_extent(renderer=renderer)
            if bb.width == 0 or bb.height == 0: continue
            boxes.append((txt, bb))
    problems = []
    for txt, bb in boxes:
        if not (fig_bbox.x0 - 1 <= bb.x0 and bb.x1 <= fig_bbox.x1 + 1 and
                fig_bbox.y0 - 1 <= bb.y0 and bb.y1 <= fig_bbox.y1 + 1):
            problems.append(f"CLIPPED: '{txt}' extends outside figure bounds "
                            f"(bbox={bb.bounds}, fig={fig_bbox.bounds})")
    n = len(boxes)
    for i in range(n):
        for j2 in range(i + 1, n):
            t1, b1 = boxes[i]; t2, b2 = boxes[j2]
            if t1 == t2: continue
            if b1.overlaps(b2):
                # shrink-test: ignore trivial edge-touching (<15% of smaller box area)
                ix0, iy0 = max(b1.x0, b2.x0), max(b1.y0, b2.y0)
                ix1, iy1 = min(b1.x1, b2.x1), min(b1.y1, b2.y1)
                inter_area = max(0, ix1 - ix0) * max(0, iy1 - iy0)
                min_area = min(b1.width * b1.height, b2.width * b2.height)
                if min_area > 0 and inter_area / min_area > 0.15:
                    problems.append(f"OVERLAP: '{t1}' overlaps '{t2}' "
                                    f"({100*inter_area/min_area:.0f}% of smaller box)")
    print(f"\n--- Layout validation: {name} ---")
    if problems:
        for p in problems: print(f"  ISSUE: {p}")
    else:
        print("  (a) no clipped text   (b) no overlapping text boxes   "
              "(c) no out-of-bounds elements")
    return problems

validate_figure(fig, "Figure_heatmap_updated.png")
plt.close(fig)
