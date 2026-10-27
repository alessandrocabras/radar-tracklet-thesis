# Local preparation checks

Prepared 2026-10-27. These checks concern the code-only release, not benchmark
performance and not verification of all historical chapter claims.

| Check | Local outcome |
| --- | --- |
| Core contract tests | 18 passed |
| Optional TensorFlow model-test module | Skipped: TensorFlow not installed in the local preparation environment |
| Exact source-function provenance | 16 extracted functions match their recorded source SHA-256 hashes |
| Three notebooks, default flags | All code cells execute without accessing private data or starting training |
| Private benchmark evaluation/training | Not executed; no benchmark, checkpoints or scalers included |
| Historical tables, figures, hyperparameter-search logs | Not reproduced; see chapter-alignment limits |

Core tests cover first/last target alignment, feature dimensions, clean-value
independence, future-observation perturbations, range baseline/error identity,
streaming padding and support selection, fixed-batch omissions/reset events,
circular versus ordinary errors, pooled metrics, training-only scaler fitting
and preservation of supplied checkpoint scalers.

Relevant local versions: Python 3.11, NumPy 2.4.6, pandas 2.2.2,
scikit-learn 1.6.1, joblib 1.5.3, pytest 8.3.5. The preparation environment is
separate from the production Colab/container environment. These versions do
not assert a historical training environment. TensorFlow tests should be run
with the neural extra in the production-compatible container before claiming
neural runtime validation of this refactored release.

Git is initialized locally with a new independent history, no remote and no
public upload. The release checker enforces an explicit file allowlist,
parses Python/notebook code and rejects saved notebook outputs. It is not a
substitute for reviewing the staged diff and institutional publication rights.
