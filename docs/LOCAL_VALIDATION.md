# Local verification

Updated 2026-10-30. These are software checks, not new benchmark results.

| Check | Outcome |
| --- | --- |
| Complete test suite, including TensorFlow | 57 passed |
| Three notebooks with default flags | All code cells execute without private access or training |
| Release checker | 28 allowlisted files; syntax and clean notebook outputs pass |
| Whitespace check | git diff --check passes |
| Private/public geometric formula comparison | Eight toy sequences, both degrees and all four lookbacks; maximum absolute Cartesian delta 9.1e-13 m |
| Full production training or dataset-wide checkpoint re-evaluation | Not rerun during this repository update |

Tests cover causal features/baselines, clean independence, target alignment,
original manifest reading, timestamps, training-only scaling, padding versus
metric masks, batch omissions and resets, architecture shapes/counts,
recurrent chunk invariance, a complete 20-epoch tiny-model control test,
checkpoint selection, frozen window/stream entry points, exact common support,
P99/extreme exports, fallback gates, seed statistics, spatial metrics,
explicit UKF units and title-free plotting.

Tests use arbitrary arithmetic/random arrays, not orbital trajectories.
The 20-epoch test uses a four-unit GRU, not the benchmark's 256-unit network.
Frozen-entry tests save/reload toy checkpoints and scalers in temporary external
directories; no private benchmark data or artifact is copied into the release.

Local Windows software: Python 3.11, NumPy 2.0.2, pandas 2.2.2,
scikit-learn 1.6.1, joblib 1.5.3, TensorFlow 2.18.0, Keras 3.8.0,
matplotlib 3.11.1, pytest 8.3.5. Pytest was reused from a separate local test
environment after the primary environment's site-packages. A fresh dependency
installation and production GPU rerun were not performed. Passing tests do not
guarantee bitwise historical reproduction.

Original source hashes in SOURCE_PROVENANCE.json identify pre-packaging
functions, not current files after refactoring. The geometric comparison
executed only the two already-inspected pure measurement-domain reference
functions, not a simulator or private experiment engine.

The existing origin points to the author's reviewer repository. This update
does not commit, push, alter remote history, or publish private artifacts.
The author must review the staged diff and institutional code-publication
rights before uploading.
