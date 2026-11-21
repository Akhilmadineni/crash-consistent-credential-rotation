# Crash-Consistent Credential Rotation Across Databases, Secret Stores, and Application Clients

This repository contains the experiment code, synthetic dataset, and stored numerical results and figures for **GRU-CR**.

## Study and data

GRU-CR encodes each run in six canonical event positions with step status and local signals, initialises the GRU from static context, and predicts inconsistency with a binary head and an auxiliary final-state head. It compares against fixed-order rules, logistic regression and a multilayer perceptron.

The dataset is **entirely synthetic**: 13,000 records and 21 columns, generated with seed 2081. It does not contain real credentials, people, companies, IP addresses, or CVEs. See [`data/A5P5_README.md`](data/A5P5_README.md) for the data dictionary and generation assumptions.

The supplied dataset README refers to `gen_author5.py`, but that generator script is not included in the supplied project. The included CSV supports rerunning the experiments; regenerating the raw dataset from scratch is outside this package.

## Repository contents

- `code/run_all.py`: complete experiment pipeline.
- `code/README.md`: method details and output descriptions.
- `requirements.txt`: Python package requirements, copied unchanged from `code/requirements.txt`.
- `data/A5P5_credential_rotations.csv`: synthetic input dataset.
- `data/A5P5_README.md`: data dictionary and synthetic generation assumptions.
- `results/`: stored result CSVs, JPG figures, and `metrics.json`.

The code uses paths relative to its own location. Keep `code/`, `data/`, and `results/` as sibling directories. Upload these contents at the repository root.

## Requirements

Use Python 3.10 or newer and an isolated virtual environment. Graphviz (`dot`) is required for the complete pipeline, which regenerates the methodology flowchart. PyTorch can use a CPU build. Install the system Graphviz program separately when needed; it is not installed by `pip`.

The supplied requirements specify minimum package versions rather than a locked environment. Exact numeric values and runtime can vary with dependency versions, platform, and hardware.

## Reproduce the experiment

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python code/run_all.py
```

On Windows, activate the environment with `.venv\Scripts\activate` before running the same Python commands. If the Graphviz step is required, check that `dot -V` succeeds first.

The complete run takes about 6 minutes on two CPU cores, according to the supplied code README. It rewrites files in `results/` and creates `figures/` with PNG outputs. Existing result files are included for inspection without rerunning model training.

Evaluation uses stratified five-fold cross-validation, with 15% of each training fold held out for early stopping and threshold selection; seeds are 2026 plus the fold index. Preprocessing is fitted inside the training data, and validation data select decision thresholds and other tuned settings. Identifiers and post-outcome impact columns do not enter the prediction inputs. The final-state category is an auxiliary training label only; it is not an inference input.

## Stored results

The stored five-fold results report F1 0.819 and PR-AUC 0.912 for GRU-CR, compared with 0.806 and 0.897 for the MLP. The protocol counterfactual map is model-derived and is not a causal estimate.

Main outputs are `results/table1_dataset_summary.csv`, `results/table3_main_comparison.csv`, `results/table4_ablation.csv`, and `results/metrics.json`. The remaining CSVs contain per-split metrics and supporting analyses. JPG figures accompany the tables. Timing measurements are environment-dependent.

## Limitations

The synthetic generator emphasises interactions involving protocol and crash position. Each run contains at most one crash. The failed-transaction coverage metric can be dominated by a few high-impact runs, and the counterfactual map is not causal.

The stored results come from synthetic experiments and do not establish production effectiveness. The source project did not supply a fully pinned environment. This package preserves the supplied code and data, with repository documentation added for reproducibility.

## Author and license

Author: Akhil Venkata Sai Madineni. This package is distributed under the [MIT license](LICENSE).

## Citation and archive

Source repository: [https://github.com/Akhilmadineni/crash-consistent-credential-rotation](https://github.com/Akhilmadineni/crash-consistent-credential-rotation).

Use [`CITATION.cff`](CITATION.cff) for the software citation. The matching [`.zenodo.json`](.zenodo.json) supplies the archive metadata for release `v1.0.0`. Version `v1.0.0` is archived at [10.5281/zenodo.24411026](https://doi.org/10.5281/zenodo.24411026). Use this version DOI to cite the fixed experiment archive.

## Data and code availability

The code, synthetic data, and results are available at https://github.com/Akhilmadineni/crash-consistent-credential-rotation, with version v1.0.0 archived at https://doi.org/10.5281/zenodo.24411026.
