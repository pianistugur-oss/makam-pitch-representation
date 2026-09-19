"""Publication figure: corpus pitch inventories of the three makams on a 53-comma ruler, with 12-TET projection and conflation-zone annotations."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import sqlite3, re

BASE       = Path("/Users/ugurozalp/makam_beklenti")   # local IDyOM output location; edit for your setup
SYMBTR_DIR = Path("/Users/ugurozalp/Downloads/SymbTr-master/txt")   # local SymbTr checkout; edit for your setup
OUTDIR     = BASE / "data" / "idyom_output"
FIGOUT     = Path(__file__).resolve().parent.parent.parent / "figures" / "Figure_makam_structure.png"

MAKAMS  = ["ussak", "huseyni", "nihavent"]
LABELS  = {"ussak": "Uşşak", "huseyni": "Hüseyni", "nihavent": "Nihâvend"}
DATASET = {"ussak": 200, "huseyni": 202, "nihavent": 204}
N_MIN   = 50

TET_POS   = [i * 53 / 12 for i in range(13)]       # comma positions of 12-TET grid
IVNAMES   = ['P1','m2','M2','m3','M3','P4','A4',
             'P5','m6','M6','m7','M7','P8']

# ── helpers ────────────────────────────────────────────────────────────────────

def _int(s):
    try: return int(s)
    except: return None

def koma_to_tet(k): return round(int(k) * 12 / 53)

SUB_TBL = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
def nota53_short(nota):
    """'Si4b1' → 'Si♭₁', 'La4' → 'La', 'Fa4#3' → 'Fa♯₃'"""
    m = re.match(r'^([A-Za-z]+)(\d+)(b|#)?(\d+)?$', nota.strip())
    if not m: return nota
    note, _, acc, cnt = m.groups()
    s = note
    if acc == 'b':  s += '♭'
    elif acc == '#': s += '♯'
    if cnt: s += cnt.translate(SUB_TBL)
    return s

def get_karar(makam):
    db = OUTDIR / makam / "koma" / "idyom.db"
    with sqlite3.connect(db) as c:
        rows = c.execute(
            "SELECT COMPOSITION_ID, CPITCH FROM mtp_event "
            "WHERE DATASET_ID=? ORDER BY COMPOSITION_ID, ONSET",
            (DATASET[makam],)).fetchall()
    last = {}
    for cid, cp in rows: last[cid] = cp
    return last

def read_dat_cp(path):
    """Read (mid, cp) skipping NA rows."""
    path = Path(path)
    with open(path) as f: hdr = f.readline().split()
    mid_i = hdr.index("melody.id")
    cp_i  = hdr.index("cpitch")
    ic_i  = hdr.index("cpitch.ic")
    rows  = []
    with open(path) as f:
        f.readline()
        for line in f:
            pts = line.split()
            if pts[ic_i] == "NA": continue
            rows.append((int(pts[mid_i]), int(pts[cp_i])))
    return rows

def get_dat_path(makam):
    return sorted((OUTDIR / makam / "koma").glob("*.dat"))[0]

# ── load corpus ────────────────────────────────────────────────────────────────

print("Loading corpus …")
DEG_COUNTS = {}
DEG_KOMAS  = {}
KOMA_NOTAS = defaultdict(Counter)

for m in MAKAMS:
    karar = get_karar(m)
    rows  = read_dat_cp(get_dat_path(m))
    cnt   = Counter()
    kdeg  = defaultdict(Counter)
    for mid, cp in rows:
        k = karar.get(mid - 1)
        if k is None: continue
        d = (cp - k) % 53
        cnt[d] += 1
        kdeg[d][cp] += 1
    DEG_COUNTS[m] = cnt
    DEG_KOMAS[m]  = kdeg
    n_qual = sum(1 for v in cnt.values() if v >= N_MIN)
    print(f"  {LABELS[m]:12s}  {sum(cnt.values())} notes  "
          f"{n_qual} degrees with n≥{N_MIN}")

print("Loading Nota53 names …")
for m in MAKAMS:
    for txt in SYMBTR_DIR.glob(f"{m}--*.txt"):
        with open(txt, encoding="utf-8") as f:
            for line in f:
                pts = line.rstrip("\n").split("\t")
                if len(pts) < 5 or _int(pts[1]) != 9: continue
                k = _int(pts[4]); n = pts[2].strip()
                if k and k > 0 and n: KOMA_NOTAS[k][n] += 1

def primary_nota(makam, deg, oct=4):
    notas = Counter()
    for koma, cnt in DEG_KOMAS[makam][deg].items():
        for nota, n in KOMA_NOTAS[koma].items():
            notas[nota] += n
    if not notas: return ""
    prefer = {n: c for n, c in notas.items() if str(oct) in n}
    best = max(prefer, key=prefer.get) if prefer else max(notas, key=notas.get)
    return nota53_short(best)

# ── degree label definitions ───────────────────────────────────────────────────

# English functional labels for theoretically named degrees
FUNC = {
    ("ussak",    0):  "tonic",
    ("huseyni",  0):  "tonic",
    ("nihavent", 0):  "tonic",
    ("ussak",    7):  "kürdi",
    ("huseyni",  7):  "kürdi",
    ("ussak",    8):  "segah",
    ("huseyni",  8):  "segah",
    ("huseyni",  9):  "nat. 2nd",    # Si natural — same 12-TET as kürdi/segah!
    ("nihavent", 17): "neut. 3rd",
    ("nihavent", 18): "maj. 3rd",
    ("ussak",    22): "güçlü",
    ("huseyni",  22): "güçlü",
    ("nihavent", 22): "perf. 4th",
    ("nihavent", 31): "dominant",
}

def deg_label(makam, deg):
    nota = primary_nota(makam, deg)
    func = FUNC.get((makam, deg))
    if func:
        return f"{func}\n({nota})" if nota and nota not in func else func
    return nota

# ── colour encoding ────────────────────────────────────────────────────────────

C_TONIC = '#111111'
C_KURDI = '#D06000'
C_SEGAH = '#BB1111'
C_STD   = '#444466'

def deg_color(makam, deg):
    if deg == 0: return C_TONIC
    if makam in ('ussak', 'huseyni') and deg == 7:  return C_KURDI
    if makam in ('ussak', 'huseyni') and deg == 8:  return C_SEGAH
    if makam == 'nihavent'           and deg == 17: return C_KURDI
    if makam == 'nihavent'           and deg == 18: return C_SEGAH
    return C_STD

def deg_size(count, max_count):
    """Marker area (pts²) for scatter; log-scaled."""
    if count < N_MIN: return 0
    lo, hi = np.log(N_MIN), np.log(max_count)
    v = np.log(count)
    return 25 + (v - lo) / max(hi - lo, 1e-9) * 420

# ── figure setup ───────────────────────────────────────────────────────────────

plt.rcParams.update({
    'font.family':      'sans-serif',
    # DejaVu Sans first: always available in matplotlib, full Unicode (♭ ♯ → ∝)
    'font.sans-serif':  ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size':        8,
    'axes.titlesize':   10,
    'xtick.labelsize':  7,
    'figure.dpi':       300,
})

fig = plt.figure(figsize=(13, 10.6), dpi=300)
gs  = fig.add_gridspec(
    5, 1,
    height_ratios=[0.32, 1.0, 1.0, 1.0, 1.05],
    hspace=0.65,
    left=0.10, right=0.96, top=0.92, bottom=0.10
)

ax_ruler    = fig.add_subplot(gs[0])
ax_ussak    = fig.add_subplot(gs[1], sharex=ax_ruler)
ax_huseyni  = fig.add_subplot(gs[2], sharex=ax_ruler)
ax_nihavent = fig.add_subplot(gs[3], sharex=ax_ruler)
ax_comp     = fig.add_subplot(gs[4])

MAQAM_AXES = {"ussak": ax_ussak, "huseyni": ax_huseyni, "nihavent": ax_nihavent}
XLIM = (-0.5, 53.5)   # tight to actual data range (0-53 commas)

# ── 12-TET ruler strip ─────────────────────────────────────────────────────────
# Note: axis positions are in COMMA units (same as the main x-axis below),
# NOT cents -- labelled with plain numbers to avoid the previous "¢" mislabel.

ax_ruler.set_xlim(*XLIM)
ax_ruler.set_ylim(0, 1)
ax_ruler.axis('off')

for i, pos in enumerate(TET_POS):
    lw = 1.2 if i in (0, 12) else 0.7
    ax_ruler.axvline(pos, color='#2255AA', linestyle='--', linewidth=lw, alpha=0.75)
    ax_ruler.text(pos, 0.55, IVNAMES[i], ha='center', va='center',
                  fontsize=6.5, color='#2255AA', fontweight='bold')
    ax_ruler.text(pos, 0.08, f"{pos:.1f}", ha='center', va='bottom',
                  fontsize=5.5, color='#3366BB')

# NOTE: ax_ruler has axis('off'), so its spine-based ylabel anchor is
# undefined/inconsistent for bbox purposes -- use a plain axes-fraction text
# instead (robust regardless of axis visibility).
ax_ruler.text(-0.02, 0.5, "12-TET\ngrid", transform=ax_ruler.transAxes,
              ha='right', va='center', fontsize=7, color='#2255AA', style='italic')

# ── helper: draw tet lines on ax ───────────────────────────────────────────────

def draw_tet_lines(ax):
    for i, pos in enumerate(TET_POS):
        lw  = 0.9 if i in (0, 12) else 0.55
        col = '#111111' if i in (0, 12) else '#2255AA'
        ax.axvline(pos, color=col, linestyle='--', linewidth=lw, alpha=0.22, zorder=0)

# ── draw conflation bracket ────────────────────────────────────────────────────

def draw_bracket(ax, d_lo, d_hi, tet_comma_pos, color, y_bracket=-1.25):
    """
    Horizontal bracket at y_bracket spanning d_lo-d_hi, short arrow toward
    tet_comma_pos. The arrow's rise is kept SHORT (contained well below the
    bracket line) so it never reaches up into the degree-label band above --
    it only needs to gesture "upward, toward the grid", not physically touch
    the dashed 12-TET line (which is independently visible as its own guide).
    """
    mid = (d_lo + d_hi) / 2
    # horizontal bar
    ax.plot([d_lo - 0.35, d_hi + 0.35], [y_bracket, y_bracket],
            color=color, lw=0.9, solid_capstyle='round', zorder=5)
    # verticals
    for x in [d_lo - 0.35, d_hi + 0.35]:
        ax.plot([x, x], [y_bracket, y_bracket + 0.13],
                color=color, lw=0.9, zorder=5)
    # short arrow, tip stays close to the bracket -- never crosses into the
    # degree-label band (kept < 0.15 units above the bracket line)
    arrow_dx = 0.5 if tet_comma_pos >= mid else -0.5
    ax.annotate('',
                xy=(mid + arrow_dx, y_bracket + 0.14),
                xytext=(mid, y_bracket - 0.08),
                arrowprops=dict(arrowstyle='->', color=color, lw=0.9),
                zorder=6)
    # label below bracket
    ax.text(mid, y_bracket - 0.20, "same 12-TET\nclass",
            ha='center', va='top', fontsize=5.8, color=color,
            multialignment='center')

# ── label placement helper ─────────────────────────────────────────────────────

def compute_sides(qualified_degs):
    """
    Assign above(+1)/below(-1) to minimise crowding.
    Dense regions (gap < 6 commas): strictly alternate.
    Sparse regions: prefer above, reset the alternation.
    Always place tonic (d=0) above.
    """
    degs = sorted(qualified_degs)
    sides = {}
    prev_d, prev_s = -99, 1   # start with previous = above so first gets below
    for d in degs:
        if d == 0:
            sides[d] = 1; prev_d, prev_s = d, 1; continue
        gap = d - prev_d
        if gap < 5:
            s = -prev_s          # strict alternation in dense zone
        elif gap < 9:
            s = -prev_s          # still alternate in moderately close zone
        else:
            s = 1                # far enough: restart above
        sides[d] = s
        prev_d, prev_s = d, s
    return sides

# ── main panel drawing ─────────────────────────────────────────────────────────

REPORT = []

for m in MAKAMS:
    ax = MAQAM_AXES[m]
    ax.set_xlim(*XLIM)
    ax.set_ylim(-2.5, 2.1)
    ax.set_yticks([])
    for sp in ('top','left','right'): ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.6)
    draw_tet_lines(ax)
    ax.axhline(0, color='#999999', lw=0.6, zorder=1)

    # Maqam name label on left (native ylabel -> plays correctly with
    # bbox_inches='tight' margin calculation, unlike a data-coordinate hack)
    ax.set_ylabel(LABELS[m], rotation=0, ha='right', va='center',
                  fontsize=11, fontweight='bold', labelpad=8)

    max_cnt  = max((v for d, v in DEG_COUNTS[m].items() if v >= N_MIN), default=1)
    qualified = sorted([(d, v) for d, v in DEG_COUNTS[m].items() if v >= N_MIN])
    sides     = compute_sides([d for d, _ in qualified])

    # scatter + labels
    for d, cnt in qualified:
        col  = deg_color(m, d)
        sz   = deg_size(cnt, max_cnt)
        side = sides.get(d, 1)
        y_mk = 0

        ax.scatter(d, y_mk, s=sz, color=col, alpha=0.88, zorder=3,
                   edgecolors='white', linewidths=0.45)

        # thin line connecting marker to label
        y_lbl_base = 0.60 * side
        ax.plot([d, d], [y_mk + 0.12 * side, y_lbl_base * 0.72],
                color=col, lw=0.4, alpha=0.5, zorder=2)

        label = deg_label(m, d)
        va    = 'bottom' if side > 0 else 'top'
        y_txt = y_lbl_base + 0.08 * side
        ax.text(d, y_txt, label, ha='center', va=va,
                fontsize=6.0, color=col, zorder=4, multialignment='center',
                linespacing=1.2)

        # report entry
        tet = round(d * 12 / 53)
        REPORT.append({"makam": LABELS[m], "d": d, "count": cnt,
                       "nota53": primary_nota(m, d),
                       "label": label.replace('\n', ' '),
                       "12tet_semitone": tet,
                       "12tet_comma_pos": tet * 53/12})

    # ── conflation brackets ────────────────────────────────────────────────────
    tet_class_degs = defaultdict(list)
    for d, cnt in qualified:
        tet = round(d * 12 / 53)
        tet_class_degs[tet].append(d)

    print(f"\n  {LABELS[m]}: degree -> MIDI(relative-semitone) class assignment")
    for tet_cls in sorted(tet_class_degs):
        ds = sorted(tet_class_degs[tet_cls])
        flag = "  <-- CONFLATED (>=2 degrees share this class)" if len(ds) >= 2 else ""
        print(f"    relative semitone {tet_cls:2d} (center={tet_cls*53/12:5.2f} commas, "
              f"round-boundary=[{(tet_cls-0.5)*53/12:5.2f},{(tet_cls+0.5)*53/12:5.2f}]): "
              f"degrees {ds}{flag}")

    special_pair = {7, 8, 9} if m in ('ussak', 'huseyni') else {17, 18}

    for tet_cls, degs in tet_class_degs.items():
        if len(degs) < 2: continue
        d_lo, d_hi    = min(degs), max(degs)
        tet_comma_pos = tet_cls * 53 / 12
        is_special    = bool(set(degs) & special_pair)
        col   = '#CC2200' if is_special else '#666666'
        # extra clearance for the special pair: its labels sit close on
        # either side (one above, one below the axis), so push its bracket
        # further down to keep the short arrow clear of the label band
        y_br  = -1.65 if is_special else -1.55
        draw_bracket(ax, d_lo, d_hi, tet_comma_pos, col, y_bracket=y_br)

    # x-axis ticks
    ax.set_xticks(range(0, 54, 5))
    ax.tick_params(axis='x', length=3, pad=2)
    if m == 'nihavent':
        ax.set_xlabel("Scale degree (commas from tonic)", fontsize=9, labelpad=4)

# ── comparison strip ───────────────────────────────────────────────────────────

ax_comp.set_xlim(-0.5, 14)
ax_comp.set_ylim(-2.7, 2.7)
for sp in ('top','right','left'): ax_comp.spines[sp].set_visible(False)
ax_comp.set_yticks([])
ax_comp.set_xticks(range(0, 14))
ax_comp.tick_params(axis='x', labelsize=7)
ax_comp.set_xlabel("Commas from tonic  [zoomed: 2nd-degree region]",
                   fontsize=9, labelpad=4)
ax_comp.axhline(0, color='#999999', lw=0.6)

m2 = TET_POS[1]   # ~4.42 commas: CENTRE of the m2 (MIDI 70) semitone
M2 = TET_POS[2]   # ~8.83 commas: CENTRE of the M2 (MIDI 71) semitone
semitone = 53 / 12
# Rounding BOUNDARIES (not centres!) delimit which comma values belong to
# which 12-TET class: a comma value rounds to class k iff it falls within
# [center_k - semitone/2, center_k + semitone/2].
b_m1_m2 = (0     + m2) / 2      # boundary between MIDI 69 (tonic) and 70
b_m2_M2 = (m2    + M2) / 2      # boundary between MIDI 70 and 71  = 6.625
b_M2_m3 = (M2 + TET_POS[3]) / 2 # boundary between MIDI 71 and 72

print(f"\n  Comparison-strip conflation zone (corrected to ROUNDING boundaries):")
print(f"    MIDI 70 (m2) band: [{b_m1_m2:.3f}, {b_m2_M2:.3f}] commas")
print(f"    MIDI 71 (M2) band: [{b_m2_M2:.3f}, {b_M2_m3:.3f}] commas  <- kürdi(7), segah(8), nat.2nd(9) all fall here")

# Shade the MIDI-71 rounding band (kürdi + segah + natural 2nd all conflate here)
ax_comp.axvspan(b_m2_M2, b_M2_m3, color='#FFF0CC', alpha=0.7, zorder=0)
# Lightly shade the adjacent MIDI-70 band too, for visual contrast/context
ax_comp.axvspan(b_m1_m2, b_m2_M2, color='#EAF1FB', alpha=0.6, zorder=0)

# 12-TET semitone CENTRES (dashed lines) + rounding BOUNDARY (dotted) for clarity
for pos, name in [(m2, "12-TET m2\n(B♭, MIDI 70)"), (M2, "12-TET M2\n(B♮, MIDI 71)")]:
    ax_comp.axvline(pos, color='#2255AA', ls='--', lw=0.9, alpha=0.8)
    ax_comp.text(pos, 2.65, name, ha='center', va='top',
                 fontsize=6.5, color='#2255AA', multialignment='center')
ax_comp.axvline(b_m2_M2, color='#AA5500', ls=':', lw=0.9, alpha=0.7)
ax_comp.text(b_m2_M2, -2.35, "rounding\nboundary", ha='center', va='top',
             fontsize=5.3, color='#AA5500', multialignment='center')

# Degree marks (middle band: labels y~0.9-1.3, markers y=0)
# d=9 now sits INSIDE the corrected MIDI-71 band -> relabelled to show it
# also conflates here (it is Huseyni's "natural 2nd", Si natural)
marks = [
    (4,  "min. 2nd\n(d=4–5)",       '#888888', 'o', 60),
    (7,  "kürdi\n(d=7)",            C_KURDI,   's', 100),
    (8,  "segah\n(d=8)",            C_SEGAH,   's', 100),
    (9,  "nat. 2nd\n(d=9, Hüseyni)", '#2255AA', 'o', 60),
]
for d, lbl, col, mk, sz in marks:
    ax_comp.scatter(d, 0, s=sz, color=col, marker=mk, zorder=3,
                    edgecolors='white', lw=0.5)
    ax_comp.text(d, 0.35, lbl, ha='center', va='bottom',
                 fontsize=6.5, color=col, multialignment='center')

# Interval-span arrows (lower band, well separated from each other:
# ~m2 at y=-0.55, ~whole tone at y=-1.15, caption at y=-1.85)
ax_comp.annotate('', xy=(m2, -0.55), xytext=(4, -0.55),
                 arrowprops=dict(arrowstyle='<->', color='#888888', lw=0.7))
ax_comp.text(2, -0.80, "~m2\n(90¢)", ha='center', va='top',
             fontsize=5.8, color='#888888', multialignment='center')

ax_comp.annotate('', xy=(M2, -1.05), xytext=(m2, -1.05),
                 arrowprops=dict(arrowstyle='<->', color='#888888', lw=0.7))
ax_comp.text((m2+M2)/2, -1.20, "~whole tone (200¢)", ha='center', va='top',
             fontsize=5.8, color='#888888')

ax_comp.text((b_m2_M2 + b_M2_m3)/2, -1.55,
             "Kürdi (d=7), segah (d=8) AND natural 2nd (d=9)\nall conflate to MIDI 71 (B♮)",
             ha='center', va='top', fontsize=6.8, color='#AA5500',
             style='italic', multialignment='center')

ax_comp.set_title(
    "Comparison: neutral 2nd (kürdi/segah) vs. Western semitone positions",
    fontsize=8, pad=4, color='#333333')

# ── legend ─────────────────────────────────────────────────────────────────────

legend_els = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor=C_TONIC,
           markersize=8, label='Tonic (karar)'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor=C_KURDI,
           markersize=8, label='Kürdi — lower neutral 2nd (d=7)'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor=C_SEGAH,
           markersize=8, label='Segah — upper neutral 2nd (d=8)'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor=C_STD,
           markersize=5, label='Other scale degree  (area proportional to log corpus frequency)'),
    Line2D([0],[0], ls='--', color='#2255AA', lw=0.9,
           label='12-TET semitone positions'),
    mpatches.Patch(facecolor='#FFF0CC', edgecolor='#AA5500',
                   label='Conflation zone (two comma pitches share the same 12-TET class)'),
]

fig.legend(handles=legend_els, loc='lower center', ncol=2,
           fontsize=7.5, frameon=True, framealpha=0.9,
           bbox_to_anchor=(0.5, -0.005), fancybox=True,
           edgecolor='#cccccc')

# ── title ──────────────────────────────────────────────────────────────────────

fig.suptitle(
    "Corpus Pitch Inventories of Three Turkish Makams on a 53-Comma Ruler\n"
    "with 12-TET Projection",
    fontsize=12, fontweight='bold', y=0.97
)

# ── save ───────────────────────────────────────────────────────────────────────

plt.savefig(FIGOUT, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nSaved: {FIGOUT}")

# ── programmatic overlap / clipping check ─────────────────────────────────────

def validate_figure(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_bbox = fig.bbox
    boxes = []
    for ax in fig.axes:
        candidates = list(ax.texts) + [ax.title, ax.xaxis.label, ax.yaxis.label]
        candidates += list(ax.get_xticklabels()) + list(ax.get_yticklabels())
        for t in candidates:
            txt = t.get_text()
            if not txt: continue
            bb = t.get_window_extent(renderer=renderer)
            if bb.width == 0 or bb.height == 0 or not np.isfinite([bb.x0,bb.x1,bb.y0,bb.y1]).all():
                continue
            boxes.append((txt, bb))
    problems = []
    for txt, bb in boxes:
        if not (fig_bbox.x0 - 1 <= bb.x0 and bb.x1 <= fig_bbox.x1 + 1 and
                fig_bbox.y0 - 1 <= bb.y0 and bb.y1 <= fig_bbox.y1 + 1):
            problems.append(f"CLIPPED: '{txt}' extends outside figure bounds")
    n = len(boxes)
    for i in range(n):
        for j2 in range(i + 1, n):
            t1, b1 = boxes[i]; t2, b2 = boxes[j2]
            if t1 == t2: continue
            if b1.overlaps(b2):
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

validate_figure(fig, "Figure_makam_structure.png")

# ── degree assignment report ───────────────────────────────────────────────────

print("\n" + "="*78)
print("DEGREE-TO-NAME ASSIGNMENTS  (verify correctness)")
print("="*78)
prev_m = None
for r in sorted(REPORT, key=lambda x: (x['makam'], x['d'])):
    if r['makam'] != prev_m:
        print(f"\n  {r['makam']}")
        print(f"  {'d':>4}  {'n':>7}  {'Nota53':>10}  {'12TET_st':>9}  "
              f"{'12TET_¢':>8}  label")
        print("  " + "─"*65)
        prev_m = r['makam']
    print(f"  {r['d']:4d}  {r['count']:7d}  {r['nota53']:>10}  "
          f"{r['12tet_semitone']:9d}  {r['12tet_comma_pos']:8.2f}  "
          f"{r['label']}")
