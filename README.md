# Microtonal Pitch Representation and Melodic Expectation in Turkish Maqam Music

This repository accompanies a study that examines how pitch representation resolution affects probabilistic models of melodic expectation in Turkish maqam music. Two pitch representations are derived from the same symbolic source — a 53-TET (Holder comma) encoding and a standard 12-TET encoding — and fed in parallel to the IDyOM model. The difference in information content (ΔIC = IC_koma − IC_12tet) quantifies how much expectation is carried exclusively by microtonal pitch distinctions. Three maqams are compared: Uşşak and Hüseyni (both featuring the characteristic neutral second, *nötr ikili*) and Nihavend (a diatonic maqam serving as a control). Results show that microtonal degrees, particularly d=8 (the neutral second, Si♭¹), produce significantly higher ΔIC in the microtonal maqams, and that this effect is consistent across pieces and not fully explained by note rarity. A rule-based implication–realisation model (Schellenberg 1996) is applied in parallel to contrast statistical surprise with structural expectedness.

---

## Requirements

- Python ≥ 3.10
- IDyOM 1.7 — [https://github.com/mtpearce/idyom](https://github.com/mtpearce/idyom)
- SBCL (Steel Bank Common Lisp) ≥ 2.0
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
python scripts/02_run_idyom.py       # run IDyOM (6 analyses: 3 maqams × 2 representations)
python scripts/03_ir_model.py        # apply Schellenberg (1996) IR model
python scripts/04_statistics.py      # piece-level and degree-level statistical tests
python scripts/05_figures.py         # generate all figures
```

Edit the path constants at the top of each script to match your local setup before running.

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
