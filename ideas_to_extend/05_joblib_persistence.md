# Extending Step 9: Saving and Reloading a Trained Pipeline with `joblib`

## What you're extending

Every model trained across the main notebook — the `Pipeline` from [Step 9](../scikit_learn_guide.ipynb),
the tuned `RandomForestClassifier` from Step 10, the `MLPClassifier` from Step 11 — has existed
only as a Python variable, in memory, for the lifetime of the notebook's kernel. Close the
notebook, and it's gone; you'd have to retrain from scratch to use it again. `joblib` is the
standard way scikit-learn models get saved to disk and loaded back later, in a different
Python process, without retraining.

## Why this matters

Training is often the expensive, slow part. *Using* a trained model to make a prediction is
usually cheap and fast. In a real application — a web service, a scheduled batch job, another
notebook entirely — you want to train once, save the result, and load that saved result
wherever predictions are actually needed, completely separately from the training code. This is
also how you'd hand a finished model to someone else, or deploy one into production.

`joblib` (which scikit-learn depends on internally, so it's already installed alongside it) is
preferred over Python's built-in `pickle` module for this specifically because scikit-learn
models are usually built around large NumPy arrays (learned weights, tree structures, etc.), and
`joblib` is optimised to serialise those efficiently.

## How to do it

### 1. Train and save a pipeline

```python
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000)),
])
pipeline.fit(X_train, y_train)

joblib.dump(pipeline, "iris_pipeline.joblib")
print("Saved. Test accuracy before saving:", pipeline.score(X_test, y_test))
```

This writes a single file, `iris_pipeline.joblib`, containing everything the fitted pipeline
needs: the scaler's learned mean/standard-deviation, and the logistic regression's learned
weights.

### 2. Load it back — ideally in a genuinely fresh process

To really prove this works, don't just reuse the same notebook kernel — restart your kernel
(or open a brand new Python shell / script) and run only this:

```python
import joblib
import numpy as np

loaded_pipeline = joblib.load("iris_pipeline.joblib")

# Predict on brand-new data, with no training code anywhere in this process
new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])   # a single flower's 4 measurements
prediction = loaded_pipeline.predict(new_flower)
print("Predicted class:", prediction)
print("Predicted probabilities:", loaded_pipeline.predict_proba(new_flower))
```

Notice this second process never imported `load_iris`, never called `train_test_split`, never
called `.fit()` — the loaded object already knows everything it learned the first time.

### 3. Confirm it's genuinely identical, not just "close"

```python
# Back in (or re-running) the original training process/script:
original_predictions = pipeline.predict(X_test)

loaded_pipeline = joblib.load("iris_pipeline.joblib")
loaded_predictions = loaded_pipeline.predict(X_test)

print("Predictions are identical:", np.array_equal(original_predictions, loaded_predictions))
```

This should print `True` — saving and loading changes nothing about the model's behaviour.

## Important caveats — read before using this for anything real

- **Never load a `.joblib` (or `.pickle`) file from a source you don't trust.** Both formats
  work by reconstructing arbitrary Python objects, which means a maliciously crafted file can
  execute arbitrary code the moment you call `joblib.load()` on it. Treat an untrusted model
  file exactly as you would an untrusted executable — because, functionally, it is one. Only
  load files you created yourself, or that came from a source you fully trust.
- **Version compatibility isn't guaranteed.** A pipeline saved with one version of
  scikit-learn is not guaranteed to load cleanly with a different version — internal object
  structures do change between releases. For anything beyond quick local experimentation, note
  down the `sklearn.__version__` (and `numpy.__version__`) used to save the file, and try to
  match it when loading later. `joblib.load()` will sometimes still work across nearby versions
  with a warning, but don't rely on that for production use.
- **For safer, more portable persistence**, look into
  [`skops`](https://skops.readthedocs.io/) (`pip install skops`), a library built specifically
  for scikit-learn model persistence that avoids `pickle`'s arbitrary-code-execution risk by
  using a restricted, inspectable serialisation format instead — the modern recommended
  alternative when a model file might cross trust boundaries (e.g. shared publicly, or loaded
  by a service that didn't create it).

## What to observe / think about

- Check the saved file's size with `os.path.getsize("iris_pipeline.joblib")` — then save the
  `RandomForestClassifier` from Step 10 the same way and compare. Tree ensembles store an
  entire learned tree structure per estimator, so expect a *much* larger file than the linear
  model above — a concrete way to feel the size difference between model types.
- Try `joblib.dump(pipeline, "iris_pipeline_compressed.joblib", compress=3)` and compare the
  resulting file size against the uncompressed version. What's the trade-off `compress`
  introduces (hint: think about save/load *time* versus file *size*)?
- Save the tuned model from Step 10's `GridSearchCV` — specifically
  `grid_search.best_estimator_`, not `grid_search` itself — since that's the actual fitted
  model you'd want to reuse, without also serialising the entire search history.
