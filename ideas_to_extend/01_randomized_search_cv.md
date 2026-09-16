# Extending Step 10: `RandomizedSearchCV` on a Larger Hyperparameter Grid

## What you're extending

In [Step 10](../scikit_learn_guide.ipynb) of the main notebook, we searched a small
3×3 grid (`n_estimators` × `max_depth`, 9 combinations total) two ways: a hand-rolled nested
loop, and `GridSearchCV`. Both tried **every single combination**, and both agreed on the same
best answer. That works fine for 9 combinations. It stops working once your grid gets bigger.

## Why this matters

`GridSearchCV` is *exhaustive*: it tries every possible combination of the values you give it.
That cost grows multiplicatively. Suppose you wanted to properly tune a `RandomForestClassifier`
across five hyperparameters instead of two:

| Hyperparameter | Values tried | Count |
|---|---|---|
| `n_estimators` | 50, 100, 200, 300, 500 | 5 |
| `max_depth` | None, 5, 10, 20, 30 | 5 |
| `min_samples_split` | 2, 5, 10, 20 | 4 |
| `min_samples_leaf` | 1, 2, 4, 8 | 4 |
| `max_features` | "sqrt", "log2", None | 3 |

Total combinations: `5 × 5 × 4 × 4 × 3 = 1,200`. With 5-fold cross-validation, `GridSearchCV`
would train `1,200 × 5 = 6,000` random forests. If each one takes even half a second, that's 50
minutes — for one model. Real projects routinely have grids larger than this, and models far
slower to train than a random forest.

`RandomizedSearchCV` solves this by **sampling** a fixed number of combinations (`n_iter`)
from the grid at random, instead of trying all of them. Counter-intuitively, this usually finds
a result nearly as good as the exhaustive search, for a small fraction of the cost — because in
most real hyperparameter spaces, only one or two parameters actually matter much, and random
sampling explores the *whole range* of every parameter, whereas a small grid only tests a few
fixed points.

## How to do it

### 1. Define distributions, not just lists

`RandomizedSearchCV` accepts either a fixed list (like `GridSearchCV`) or a **probability
distribution** to sample from, using `scipy.stats`. This is the real advantage over
`GridSearchCV`: instead of guessing 5 fixed values for `n_estimators`, you can say "anywhere
from 50 to 500, uniformly" and let the search explore that whole continuous range.

```python
from scipy.stats import randint, uniform

param_distributions = {
    "n_estimators": randint(50, 500),          # random integer in [50, 500)
    "max_depth": randint(2, 50),                 # random integer in [2, 50)
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 10),
    "max_features": ["sqrt", "log2", None],       # still fine to mix in fixed choices
}
```

### 2. Run the search

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import time

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=50,          # only try 50 random combinations, not all of them
    cv=5,
    random_state=42,     # makes which combinations get sampled reproducible
    n_jobs=-1,            # use all CPU cores
)

start = time.time()
random_search.fit(X_train, y_train)   # X_train, y_train from Step 4 of the main notebook
print(f"Search took {time.time() - start:.1f}s")
print("Best params:", random_search.best_params_)
print("Best cv accuracy:", random_search.best_score_)
```

With `n_iter=50` out of 1,200 possible combinations, you're only evaluating about 4% of the
full grid — but because the samples are spread across the *entire* range of every parameter
rather than clustered at a few fixed points, you'll typically land within a percent or two of
what the full `GridSearchCV` would have found.

### 3. Prove it to yourself: compare against the full grid

This is the real learning exercise — don't just take the speed claim on faith, measure it,
exactly the way the main notebook measured NumPy's vectorisation speedup and confirmed
`GridSearchCV` agreed with a manual search.

```python
from sklearn.model_selection import GridSearchCV

# Use a SMALL fixed grid here so the full exhaustive search is still feasible to run for comparison
small_grid = {
    "n_estimators": [50, 100, 200, 300, 500],
    "max_depth": [None, 5, 10, 20, 30],
}

start = time.time()
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), small_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
grid_time = time.time() - start

start = time.time()
random_search_small = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=small_grid,
    n_iter=8,   # try only 8 of the 25 combinations
    cv=5, random_state=42, n_jobs=-1,
)
random_search_small.fit(X_train, y_train)
random_time = time.time() - start

print(f"GridSearchCV:      {grid_time:.1f}s, best score = {grid_search.best_score_:.4f}")
print(f"RandomizedSearchCV: {random_time:.1f}s, best score = {random_search_small.best_score_:.4f}")
print(f"Speedup: {grid_time / random_time:.1f}x, for a score difference of "
      f"{grid_search.best_score_ - random_search_small.best_score_:+.4f}")
```

You should see `RandomizedSearchCV` finish several times faster, landing on a best score very
close to (sometimes exactly equal to) `GridSearchCV`'s result.

## What to observe / think about

- **Increase `n_iter` and watch the gap close.** At `n_iter=25` (the full grid size), a
  randomized search covering all 25 combinations is really just a shuffled grid search. Try
  `n_iter` values of 4, 8, 16, 25 and plot best score vs. `n_iter` — you'll see diminishing
  returns, which is the key insight behind why randomized search works at all.
- **`random_state` controls which combinations get tried**, not how many. Change it and rerun
  to see how much the result varies run to run with a small `n_iter` — this variance is the
  price you pay for the speedup.
- For truly large or expensive search spaces (like tuning a neural network — see
  [`04_scikeras_tuning.md`](04_scikeras_tuning.md)), `RandomizedSearchCV` with a modest `n_iter`
  is often the *only* practical option; exhaustive `GridSearchCV` simply never finishes.

## Related scikit-learn tools worth knowing about

- **`HalvingGridSearchCV` / `HalvingRandomSearchCV`** (in `sklearn.model_selection`, still
  experimental): start every candidate on a small subset of data, then progressively give more
  data/resources only to the best-performing candidates — often faster than either approach
  above for large searches.
- **`scipy.stats.loguniform`**: useful for hyperparameters that span orders of magnitude, like
  a regularisation strength `C` or a learning rate, where sampling uniformly between 0.0001 and
  100 would waste almost all samples on large values. `loguniform(1e-4, 1e2)` samples evenly
  across *orders of magnitude* instead.
