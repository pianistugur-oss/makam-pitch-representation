# Microtonal Pitch Representation and Melodic Expectation in Turkish Maqam Music

> **Note on this version.** This repository reflects the first revision of the manuscript for *Music Perception*. It adds ten analyses (scripts `00`, `08`–`16`) requested during review, alongside the original pipeline (`01`–`07`). Raw IDyOM `.dat` output files, resampling caches, and the SymbTr corpus itself are **not** included in this repository (see `.gitignore`); every number in `results/` is fully reproducible by running the pipeline from `01_preprocess.py` onward against a local SymbTr checkout and a local IDyOM installation.

This repository accompanies a study that examines how pitch representation resolution affects probabilistic models of melodic expectation in Turkish maqam music. Two pitch representations are derived from the same symbolic source — a 53-TET (Holder comma) encoding and a standard 12-TET encoding — and fed in parallel to the IDyOM model. The difference in information content (ΔIC = IC_koma − IC_12tet) quantifies how much expectation is carried exclusively by microtonal pitch distinctions. Three makams are compared: Uşşak and Hüseyni (both featuring the characteristic neutral second, *nötr ikili*) and Nihâvend (a diatonic makam serving as a control). Results show that microtonal degrees, particularly d=8 (the neutral second, Si♭¹), produce significantly higher ΔIC in the microtonal makams, and that this effect is consistent across pieces and not fully explained by note rarity. A rule-based implication–realisation model (Schellenberg 1996) is applied in parallel to contrast statistical surprise with structural expectedness.

---

## Requirements

- Python ≥ 3.10
- IDyOM 1.7 — [https://github.com/mtpearce/idyom](https://github.com/mtpearce/idyom)
- SBCL (Steel Bank Common Lisp) ≥ 2.0
- statsmodels ≥ 0.14 (used by scripts `08`, `13`)
- See `requirements.txt` for Python dependencies

## Installation

```bash
pip install -r requirements.txt
```

Install IDyOM following the official instructions at [https://github.com/mtpearce/idyom](https://github.com/mtpearce/idyom).

Set the environment variable required on macOS:

```bash
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/sqlite/lib
```

## Usage

Run the scripts in order:

```bash
python scripts/01_preprocess.py      # parse SymbTr TXT files → IDyOM SQLite databases
python scripts/02_run_idyom.py       # run IDyOM (6 analyses: 3 makams × 2 representations)
python scripts/03_ir_model.py        # apply Schellenberg (1996) IR model
python scripts/04_statistics.py      # piece-level and degree-level statistical tests
python scripts/05_figures.py         # generate all figures
python scripts/06_robustness_size_matched.py  # re-run group comparisons on size-matched
                                              # subsamples (n = 88 per makam, 5 random seeds)
                                              # to check robustness to corpus-size differences
python scripts/07_bayes_equivalence_ir.py     # compute a JZS Bayes factor (BF₀₁) and TOST
                                              # equivalence test for the implication–realization
                                              # group comparison; requires pingouin
```

Scripts 06 and 07 read from the IDyOM output files produced by scripts 01–02 and do not require re-running IDyOM. All numbered scripts write their CSV output to `results/`; figure scripts (`05_figures.py` and everything under `scripts/figures/`) write to `figures/`.

Edit the path constants at the top of each script (`BASE`, `SYMBTR_DIR`) to match your local setup before running.

### Analyses added for the first revision (Music Perception)

```bash
python scripts/00_cv_fold_verification.py       # checksum-verifies that the koma and 12-TET
                                                 # arms share identical IDyOM cross-validation
                                                 # fold partitions (prerequisite for script 09)
python scripts/08_baselines_and_diagnostics.py  # analytic bound H(comma | 12-TET),
                                                 # predictive entropy, note-level conflation
                                                 # regression, perde identification
python scripts/09_marginalization_test.py       # comma model marginalized to 12-TET classes,
                                                 # compared event-for-event with the natively
                                                 # 12-TET model
python scripts/10_permutation_control.py        # 40 permutations of comma labels within
                                                 # 12-TET classes, model retrained from scratch
                                                 # for each (~3.5 hours; use --skip-build
                                                 # --skip-run to re-analyze existing output)
python scripts/11_respelling_test.py            # counterfactual corpora with the second-degree
                                                 # region of Uşşak and Hüseyni assigned uniformly
                                                 # to segah or to kürdi (~15-20 min; same
                                                 # --skip-build/--skip-run flags apply)
python scripts/12_degree_context_analysis.py    # co-occurrence tests for high-ΔIC degrees
                                                 # against theoretically predicted companion
                                                 # degrees (Nihâvend d=36, d=17; Uşşak d=8 ref.);
                                                 # also writes segah_prevalence.csv
python scripts/13_d37_context.py                # same co-occurrence test for Hüseyni d=37,
                                                 # plus a piece-length confound check
python scripts/14_exact_statistics.py           # test statistics and exact p values reported
                                                 # in the paper (Wilcoxon, Mann-Whitney,
                                                 # Kruskal-Wallis, Cohen's d)
python scripts/15_ir_factor_decomposition.py    # decomposition of ΔIR into the five
                                                 # Schellenberg (1996) IR factors
python scripts/16_alt_12tet_mapping.py          # alternative 12-TET mapping (SymbTr NotaAE
                                                 # simple-Western respelling) vs. round(K×12/53);
                                                 # a mapping-choice robustness check
```

Scripts `00`, `08`, `09`, `12`–`16` read from existing IDyOM output (or the raw SymbTr TXT files) and do not require re-running IDyOM. Scripts `10` and `11` retrain IDyOM from scratch on counterfactual/permuted corpora they build themselves; both accept `--skip-build` and `--skip-run` to reuse databases or `.dat` output from a previous run.

Publication figure scripts live under `scripts/figures/` and read from the same IDyOM output as scripts `08`–`16`:

```bash
python scripts/figures/makam_structure_figure.py        # Figure_makam_structure.png
python scripts/figures/degree_makam_heatmap_figure.py   # Figure_heatmap_updated.png
python scripts/figures/conflation_mechanism_figure.py   # Figure_mechanism.png
python scripts/figures/baseline_comparison_figure.py    # Figure_baselines.png
python scripts/figures/piece_violin_figure.py            # pub_violin_piece_dic_eng.png (Figure 2)
```

## Results

All CSV output lives in `results/`. All figures live in `figures/`.

Two results files support specific robustness claims in the manuscript:
- **`segah_prevalence.csv`** (written by script `12`) — supports the claim that segah (d=8) is only the majority neutral-second choice in a minority of Uşşak/Hüseyni pieces (most pieces favour kürdi, d=7 by a ~6–7× note-count margin), i.e. the comma-level distinction is corpus-variable, not an artefact of one dominant piece. `segah_prevalence.csv` recomputes this statistic from the raw SymbTr corpus (118 pieces) rather than the filtered IDyOM pipeline output (117 pieces); the qualitative result is unchanged.
- **`alt_mapping.csv`** (written by script `16`) — supports the robustness-to-mapping-choice claim: an alternative, notation-derived 12-TET assignment diverges from round(Koma53×12/53) for ~12–17% of Uşşak/Hüseyni notes (mainly the d=7/8 region), and the paper's qualitative pattern is checked against this alternative rather than assumed to depend on one rounding convention.

## Data

The corpus used in this study is drawn from the **SymbTr** dataset (v2):

> Karaosmanoğlu, M. K. (2012). A Turkish makam music symbolic database for music information retrieval: SymbTr. In *Proceedings of the 13th International Society for Music Information Retrieval Conference (ISMIR)*, pp. 223–228.

SymbTr is available at: [https://github.com/MTG/SymbTr](https://github.com/MTG/SymbTr)

See `data/README.md` for setup instructions.

## Citation

If you use this code, please cite:

> [manuscript citation placeholder]

## License

MIT License. See `LICENSE` for details.
