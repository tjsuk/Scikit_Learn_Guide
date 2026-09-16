# Ideas to Extend — Detailed Guides

Each file here expands one item from `scikit_learn_guide.ipynb`'s closing "Ideas to extend"
section into a full, self-contained walkthrough: why the idea matters, a step-by-step
explanation of how to do it, complete runnable code, and things to try afterward to deepen the
understanding. They assume you've already worked through the main notebook, and reference its
variables (`X_train`, `Xd_train`, `pipeline`, etc.) directly.

| Guide | Extends | What you'll learn |
|---|---|---|
| [01_randomized_search_cv.md](01_randomized_search_cv.md) | Step 10 | Searching a hyperparameter space that's too large for exhaustive `GridSearchCV`, and proving the speed/accuracy trade-off with real timings |
| [02_column_transformer.md](02_column_transformer.md) | Step 9 | Preprocessing real-world data with mixed numeric and categorical columns in one pipeline |
| [03_run_the_keras_snippet.md](03_run_the_keras_snippet.md) | Step 12 | Actually running the illustrative Keras example, and comparing it head-to-head with Step 11's `MLPClassifier` |
| [04_scikeras_tuning.md](04_scikeras_tuning.md) | Steps 10 + 12 | Wrapping a Keras model so `GridSearchCV`/`RandomizedSearchCV` can tune it like any scikit-learn estimator |
| [05_joblib_persistence.md](05_joblib_persistence.md) | Step 9 | Saving a trained pipeline to disk and reloading it in a fresh process, plus the security caveats that come with that |
| [06_full_mnist.md](06_full_mnist.md) | Step 11 | Rerunning the capstone at ~39x the data scale, and feeling where scikit-learn's CPU-only neural network starts to strain |

## Suggested order

If you want to work through all six, this order builds on itself reasonably well:

1. **05** (persistence) — quick, low-risk, no new dependencies
2. **02** (`ColumnTransformer`) — extends preprocessing ideas you already know
3. **01** (`RandomizedSearchCV`) — extends hyperparameter search you already know
4. **06** (full MNIST) — same models as Step 11, just at real scale
5. **03** (run the Keras snippet) — introduces TensorFlow
6. **04** (SciKeras tuning) — combines 01 and 03, so do it last
