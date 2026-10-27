# Causal Radar Tracklet Forecasting — Thesis Review Companion

Code companion to the thesis chapter **Hybrid Deep Learning for Space-Debris
Radar Tracklet Reconstruction**. The implementation exposes the causal
measurement-domain task, coordinate-specific preprocessing, window-model
families, and final stateful azimuth semantics for review.

**Code-only release. No dataset, orbital simulator, split manifest, trained
weights, scalers, historical figures, or private observation exports are
distributed. This is not a full reproduction archive for the numerical
tables and figures in the chapter.**

## Start here

1. [Chapter alignment and evidence limits](docs/CHAPTER_ALIGNMENT.md).
2. [Private input interface](docs/DATA_INTERFACE.md).
3. [Source provenance](docs/SOURCE_PROVENANCE.json): source function names,
   original notebook cell indices and SHA-256 identifiers.
4. [Publication instructions](docs/PUBLISH_IT.md).

The chapter reports historical window-model results and a separate audited
stateful azimuth result. They must not be conflated. This release does not
claim that its example constructors reproduce the historical hyperparameter
search, epoch-87 checkpoint, UKF improvements or hardware timings.

## Repository layout

```text
notebooks/                 three opt-in private-data review notebooks
src/radar_tracklets/
  preprocessing.py         coordinate features, targets, causal baselines
  models.py                explicit MLP/CNN/GRU/GRU-CNN and final azimuth GRU
  streaming.py             chunking, fixed trajectory slots, resets, masks
  workflows.py             private-manifest loading and training-only scaling
  evaluation.py            physical reconstruction and pooled metrics
  training.py              opt-in final-azimuth fitting selected on validation RMSE
tests/                     arbitrary toy-array contract tests
tools/check_release.py     allowlisted-file and clean-notebook checks
docs/                      chapter mapping, provenance, limitations, publishing
```

## Install and run tests

Python 3.11 is the reference interpreter. The production audit recorded
TensorFlow 2.18.0; install the neural extra in a compatible environment.
No claim is made that dependency ranges form an exact historical lockfile.

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/container: source .venv/bin/activate
python -m pip install -e ".[test]"
python -m pytest -q
python tools/check_release.py
```

The core tests need no data or GPU. TensorFlow-specific model tests are
explicitly skipped without TensorFlow. To run them and open notebooks:

```bash
python -m pip install -e ".[neural,notebooks,test]"
python -m pytest -q
python -m jupyterlab
```

In Google Colab connected to a local container, install into the **active
kernel**, e.g. `%pip install -e /mounted/path/to/this/repository[neural,notebooks,test]`.
The notebook paths must be container-visible. None depends on a particular
Drive account, Windows username, or `/content` project directory.

## Task contract

- Canonical inputs: 25 noisy observations through t; one target at t+1.
- Elevation: direct clean-value regression, eight input features.
- Range: nine input features and residual around a past-data polynomial
  forecast, not residual around an unavailable target-time measurement.
- Canonical azimuth: six features relative to the first noisy window pivot.
- Final azimuth: nine features, GRU 256 with no convolutions, dense sequence
  outputs; Huber delta 2 on the **standardized residual** scale.
- No clean predictor features or autoregressive teacher forcing.
- Window CNNs use kernel 5 with `same` padding, within the observed window.
- Stateful azimuth uses batches of eight distinct trajectories and consecutive
  non-overlapping chunks of 25. State is reset at group boundaries, not chunks.
- Final evaluation starts at target index 25. Padding weights and metric
  selection are separate mechanisms. A complete 319-record trajectory supplies
  294 evaluated targets.
- Errors are pooled across valid target samples. Elevation is an ordinary
  difference; only azimuth is circular. Angular metrics are mrad, range is m.

## Private-data notebooks

`elevation.ipynb`, `range.ipynb` and `azimuth.ipynb` do nothing with private
data unless their opt-in flags are set. Use original private manifests and
compatible models/scalers. All fitting is explicit; examples do not save or
overwrite files. Do not commit executed notebooks containing private outputs.
Keep all private inputs and generated artefacts outside the repository.

## Availability and reuse

Dataset/simulator publication permission has not been established. Their
absence is intentional. This repository does not grant access to them.
No open-source license has been assigned automatically: before public release,
the author/institution should verify code-publication rights and choose a
license if appropriate. Public visibility alone does not grant reuse rights.

