"""Publication figure: observed ΔIC vs. two null baselines (analytic H(comma|12-TET) and 40-seed permutation), note-level, grayscale-safe."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

BASE   = Path("/Users/ugurozalp/makam_beklenti")   # unused (script has no IDyOM-output dependency); kept for reference
FIGOUT = Path(__file__).resolve().parent.parent.parent / "figures" / "Figure_baselines.png"
FIGOUT.parent.mkdir(exist_ok=True)

MAKAMS = ["ussak", "huseyni", "nihavent"]
LABELS = {"ussak": "Uşşak", "huseyni": "Hüseyni", "nihavent": "Nihâvend"}

# note-level values, cross-checked against results/baselines_and_diagnostics.csv
# (section 1A) and results/permutation_control.csv
OBSERVED = {"ussak": 0.035242, "huseyni": 0.047357, "nihavent": 0.022370}
ANALYTIC = {"ussak": 0.13778,  "huseyni": 0.15268,  "nihavent": 0.05010}

# N=40 re-trained-seed permutation null (mean + 2.5-97.5 percentile band)
PERM_MEAN = {"ussak": 0.29933, "huseyni": 0.33560, "nihavent": 0.12072}
PERM_P025 = {"ussak": 0.28714, "huseyni": 0.32195, "nihavent": 0.10965}
PERM_P975 = {"ussak": 0.30940, "huseyni": 0.34643, "nihavent": 0.13138}
PERM_LO   = {m: PERM_MEAN[m] - PERM_P025[m] for m in PERM_MEAN}
PERM_HI   = {m: PERM_P975[m] - PERM_MEAN[m] for m in PERM_MEAN}

RATIO_ANALYTIC = {m: 100 * OBSERVED[m] / ANALYTIC[m] for m in MAKAMS}
RATIO_PERM     = {m: 100 * OBSERVED[m] / PERM_MEAN[m] for m in MAKAMS}

# ── console printout for cross-checking against the figure ────────────────────
print(f"{'Maqam':10s}  {'Observed':>10}  {'Analytic':>10}  {'Permutation':>12}  "
      f"{'Obs/Analytic':>13}  {'Obs/Perm':>10}")
print("-"*75)
for m in MAKAMS:
    print(f"{LABELS[m]:10s}  {OBSERVED[m]:10.5f}  {ANALYTIC[m]:10.5f}  "
          f"{PERM_MEAN[m]:12.5f}  {RATIO_ANALYTIC[m]:12.1f}%  {RATIO_PERM[m]:9.1f}%")

# ── figure style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8,
    'figure.dpi': 300,
})

# grayscale-safe palette + hatching
GRAY_OBS  = '#1a1a1a'
GRAY_ANA  = '#888888'
GRAY_PERM = '#c8c8c8'
HATCH_OBS, HATCH_ANA, HATCH_PERM = '', '///', '...'

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 5), dpi=300)

# ── Panel (a): absolute comparison ─────────────────────────────────────────────
x = np.arange(len(MAKAMS))
w = 0.26

obs_vals  = [OBSERVED[m] for m in MAKAMS]
ana_vals  = [ANALYTIC[m] for m in MAKAMS]
perm_vals = [PERM_MEAN[m] for m in MAKAMS]
perm_err  = [[PERM_LO[m] for m in MAKAMS], [PERM_HI[m] for m in MAKAMS]]

b1 = ax_a.bar(x - w, obs_vals, width=w, color=GRAY_OBS, hatch=HATCH_OBS,
              edgecolor='black', linewidth=0.6, label='Observed')
b2 = ax_a.bar(x,     ana_vals, width=w, color=GRAY_ANA, hatch=HATCH_ANA,
              edgecolor='black', linewidth=0.6, label='Analytic baseline')
b3 = ax_a.bar(x + w, perm_vals, width=w, color=GRAY_PERM, hatch=HATCH_PERM,
              edgecolor='black', linewidth=0.6, yerr=perm_err, capsize=4,
              ecolor='black', error_kw={'linewidth': 1.0},
              label='Permutation baseline (N=40)')

for bars in (b1, b2, b3):
    for rect in bars:
        h = rect.get_height()
        ax_a.text(rect.get_x() + rect.get_width()/2, h + 0.008, f"{h:.3f}",
                   ha='center', va='bottom', fontsize=7.2)

ax_a.set_xticks(x)
ax_a.set_xticklabels([LABELS[m] for m in MAKAMS])
ax_a.set_ylabel("Mean ΔIC per note [bits]")
ax_a.set_xlim(-0.6, 2.6)
ax_a.set_ylim(0, 0.46)
ax_a.set_title("(a) Absolute comparison", fontsize=10.5, loc='left')
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)
ax_a.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3,
            frameon=False, fontsize=7.6, handlelength=1.4, columnspacing=1.0,
            handletextpad=0.5)

# ── Panel (b): ratio comparison ────────────────────────────────────────────────
w2 = 0.32
r1 = ax_b.bar(x - w2/2, [RATIO_ANALYTIC[m] for m in MAKAMS], width=w2,
              color=GRAY_ANA, hatch=HATCH_ANA, edgecolor='black', linewidth=0.6,
              label='Observed / Analytic')
r2 = ax_b.bar(x + w2/2, [RATIO_PERM[m] for m in MAKAMS], width=w2,
              color=GRAY_PERM, hatch=HATCH_PERM, edgecolor='black', linewidth=0.6,
              label='Observed / Permutation')

for bars in (r1, r2):
    for rect in bars:
        h = rect.get_height()
        ax_b.text(rect.get_x() + rect.get_width()/2, h + 1.2, f"{h:.0f}%",
                   ha='center', va='bottom', fontsize=8)

ax_b.set_xlim(-0.6, 2.6)
ax_b.axhline(100, color='black', linestyle='--', linewidth=1.0)
ax_b.text(2.55, 101.5, "arbitrary refinement", va='bottom',
          ha='right', fontsize=7.8, style='italic', color='#333333')

ax_b.set_xticks(x)
ax_b.set_xticklabels([LABELS[m] for m in MAKAMS])
ax_b.set_ylabel("Observed ΔIC as % of baseline")
ax_b.set_ylim(0, 112)
ax_b.set_title("(b) Relative to each baseline", fontsize=10.5, loc='left')
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)
ax_b.legend(loc='upper right', bbox_to_anchor=(1.0, 0.80), frameon=False, fontsize=8)

fig.text(0.5, -0.02,
          "Lower values indicate that context resolves more of the comma-choice uncertainty.",
          ha='center', va='top', fontsize=8, style='italic', color='#333333')

fig.suptitle("Observed information gain against two baselines", fontsize=13, fontweight='bold', y=1.03)

plt.tight_layout()
plt.savefig(FIGOUT, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nSaved: {FIGOUT}")
