# Extending Step 10 + Step 12: Tuning a Keras Model with `GridSearchCV` via SciKeras

## What you're extending

This combines two ideas from the main notebook: [Step 10](../scikit_learn_guide.ipynb)'s
`GridSearchCV`, which only knows how to tune scikit-learn estimators, and [Step 12](../scikit_learn_guide.ipynb)'s
observation that a Keras model can be wrapped so scikit-learn's tools work on it. **SciKeras**
is the library that performs that wrapping: it makes a Keras model *look like* a scikit-learn
`Estimator` (giving it `fit`/`predict`/`score`), so everything built around the Estimator API in
this notebook — `Pipeline`, `cross_val_score`, `GridSearchCV`, `RandomizedSearchCV` — works on a
neural network exactly as it did on `RandomForestClassifier` in Step 10.

## Why this matters

Step 11's `MLPClassifier` is easy to tune with `GridSearchCV` because it's already a native
scikit-learn estimator — you can search over `hidden_layer_sizes`, `alpha`, `learning_rate_init`,
etc. directly. A Keras model has no such built-in integration; its `fit`/`predict` methods don't
match scikit-learn's expected interface (different argument names, different return types, no
`get_params`/`set_params` for `GridSearchCV` to inspect and set hyperparameters). SciKeras
bridges that gap, so you can get the best of both worlds: Keras's architectural flexibility
(Step 12's whole point), combined with scikit-learn's hyperparameter search tooling
(Step 10's whole point).

## How to do it

### 1. Install SciKeras

```bash
pip install scikeras tensorflow
```

### 2. Write a model-building function with hyperparameters as arguments

This is the key difference from a plain Keras script: instead of hard-coding layer sizes,
write a function that *takes hyperparameters as arguments* and returns a compiled model.
SciKeras calls this function itself, using whichever hyperparameter values the search is
currently trying.

```python
from tensorflow import keras


def build_model(hidden_layer_size=64, learning_rate=0.001, meta=None):
    # `meta` is supplied automatically by SciKeras with dataset info, e.g. number of classes
    n_classes = meta["n_classes_"]
    n_features = meta["n_features_in_"]

    model = keras.Sequential([
        keras.layers.Dense(hidden_layer_size, activation="relu", input_shape=(n_features,)),
        keras.layers.Dense(hidden_layer_size // 2, activation="relu"),
        keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
```

### 3. Wrap it with SciKeras's `KerasClassifier`

```python
from scikeras.wrappers import KerasClassifier

clf = KerasClassifier(
    model=build_model,
    hidden_layer_size=64,     # default value; the search will override this
    learning_rate=0.001,       # default value; the search will override this
    epochs=30,
    verbose=0,
)
```

`clf` now behaves like any other scikit-learn classifier: `clf.fit(X, y)`, `clf.predict(X)`,
`clf.score(X, y)` all work.

### 4. Search over its hyperparameters

SciKeras exposes the arguments of `build_model` for tuning, prefixed with `model__`
(hyperparameters that affect model *architecture*), and training arguments like `epochs` or
`batch_size` directly by name (these affect how `fit` runs, not the model itself):

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "model__hidden_layer_size": [32, 64, 128],
    "model__learning_rate": [0.01, 0.001, 0.0001],
    "epochs": [20, 50],
}

grid_search = GridSearchCV(clf, param_grid=param_grid, cv=3)
grid_search.fit(Xd_train_scaled, yd_train)   # reuse Xd_train_scaled, yd_train from Step 11 / idea 03

print("Best params:", grid_search.best_params_)
print("Best cv accuracy:", grid_search.best_score_)
```

### 5. Sanity-check the cost before you run it

**Stop and calculate before running this for real.** The grid above has
`3 × 3 × 2 = 18` combinations, times `cv=3` folds = **54 total model trainings**, each running
20-50 epochs of a neural network. Compare that to Step 10's random forest search, where each
"fit" took a fraction of a second — a neural network fit takes seconds to minutes. This is
worth explicitly timing before committing to a full run:

```python
import time

start = time.time()
clf.fit(Xd_train_scaled[:200], yd_train[:200])   # a small subset, just to measure one fit's cost
one_fit_time = time.time() - start

estimated_total = one_fit_time * 54   # 54 = combinations x cv folds, from the grid above
print(f"One fit on a small subset took {one_fit_time:.1f}s")
print(f"Estimated total time for the full grid search: {estimated_total / 60:.1f} minutes")
```

This is precisely why [`01_randomized_search_cv.md`](01_randomized_search_cv.md)'s technique
matters so much *more* here than for a random forest: swap `GridSearchCV` for
`RandomizedSearchCV` with a small `n_iter` (e.g. 10) to make neural network tuning practical at
all:

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform

param_distributions = {
    "model__hidden_layer_size": [16, 32, 64, 128, 256],
    "model__learning_rate": loguniform(1e-4, 1e-1),   # see idea 01 for why loguniform fits here
    "epochs": [20, 30, 50],
}

random_search = RandomizedSearchCV(clf, param_distributions, n_iter=10, cv=3, random_state=42)
random_search.fit(Xd_train_scaled, yd_train)
print("Best params:", random_search.best_params_)
```

## What to observe / think about

- **Compare the tuned SciKeras model's accuracy against Step 11's untuned `MLPClassifier`
  (0.9639) and Step 11's tuned `GridSearchCV`-on-a-random-forest result from Step 10 (0.958).**
  Did tuning the neural network's architecture and learning rate actually buy you meaningfully
  better accuracy on this dataset, or did it land in the same ballpark regardless? For a small,
  easy dataset like the digits data, it's common for extensive tuning to yield only marginal
  gains — a useful, humbling lesson about when hyperparameter search is and isn't worth the
  computational cost.
- **Try tuning `MLPClassifier` (Step 11's native scikit-learn model) with `GridSearchCV`
  directly, without SciKeras at all**, searching over its own `hidden_layer_sizes` and
  `learning_rate_init` parameters. Since `MLPClassifier` is *already* a scikit-learn estimator,
  this needs no wrapper — compare how much simpler that setup is against the SciKeras version
  above, and how the tuned accuracy compares. This is the clearest possible illustration of
  Step 12's core message: scikit-learn's own `MLPClassifier` trades architectural flexibility
  for exactly this kind of "batteries-included" convenience.
- Look at what happens to `grid_search.best_estimator_` — it's a fitted `KerasClassifier`, and
  `grid_search.best_estimator_.model_` gives you the underlying Keras model, on which you can
  call ordinary Keras methods like `.summary()` to inspect the winning architecture directly.
