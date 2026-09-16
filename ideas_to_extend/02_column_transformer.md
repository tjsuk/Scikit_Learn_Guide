# Extending Step 9: `ColumnTransformer` for Mixed Numeric + Categorical Data

## What you're extending

[Step 9](../scikit_learn_guide.ipynb) of the main notebook introduced `Pipeline`, which chains
preprocessing and a model into one estimator — but every example in the notebook (Iris, the
income/experience demo, the digit images) was **entirely numeric**. Real-world tabular data
almost never is: a customer record might mix a numeric `age`, a numeric `income`, and a
categorical `city` or `education_level`. `StandardScaler` — the only preprocessing tool the
main notebook used — doesn't make sense applied to a text category like `"London"`.
`ColumnTransformer` is scikit-learn's answer to that: **different preprocessing for different
columns, combined into one step.**

## Why this matters

Suppose you tried to scale a categorical column with `StandardScaler`, or feed raw category
strings straight into `LogisticRegression`. Both fail outright — `StandardScaler` requires
numeric input, and scikit-learn's models expect numbers, not strings, since internally they're
doing arithmetic (weighted sums, distances, splits) on every feature. The categories first need
to become numbers, most commonly via **one-hot encoding**: turning a column with categories
`{"London", "Paris", "Berlin"}` into three separate 0/1 columns, `is_London`, `is_Paris`,
`is_Berlin`. But you generally do **not** want to one-hot encode a numeric column, or scale a
categorical one — each column type needs its own treatment. `ColumnTransformer` applies a list
of `(name, transformer, columns)` triples, each transformer only touching the columns you tell
it to, then concatenates the results into one array — all inside a single `fit`/`transform`
call, so it slots into a `Pipeline` exactly like `StandardScaler` did in the main notebook.

## How to do it

### 1. Install pandas (optional, but strongly recommended)

The main notebook deliberately avoided `pandas` to keep dependencies minimal, but mixed-type
data is genuinely awkward to work with as a plain NumPy array (NumPy arrays need one uniform
dtype for every element; a DataFrame lets each column keep its own type). Add it to your
environment:

```bash
pip install pandas
```

### 2. Build a small mixed-type dataset

```python
import pandas as pd
import numpy as np

rng = np.random.default_rng(seed=0)
n = 200

data = pd.DataFrame({
    "age": rng.integers(18, 70, n),
    "income": rng.normal(45000, 15000, n).round(0),
    "city": rng.choice(["London", "Paris", "Berlin", "Madrid"], n),
    "education": rng.choice(["High School", "Bachelors", "Masters", "PhD"], n),
})

# A made-up target: "will this person subscribe?" - loosely tied to income and education
education_boost = data["education"].map({"High School": 0, "Bachelors": 1, "Masters": 2, "PhD": 3})
score = (data["income"] - 45000) / 15000 + education_boost * 0.5 + rng.normal(0, 1, n)
target = (score > score.median()).astype(int)

print(data.head())
print("\ndtypes:\n", data.dtypes)
```

Notice `age` and `income` are numeric, `city` and `education` are categorical (`object` dtype)
— exactly the mix `ColumnTransformer` is built for.

### 3. Build column-specific preprocessing

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_features = ["age", "income"]
categorical_features = ["city", "education"]

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])
```

- **`handle_unknown="ignore"`** matters in practice: if the test data (or live production data)
  contains a category the encoder never saw during training — say a new city — the default
  behaviour is to raise an error. `"ignore"` instead encodes it as all-zeros, letting the
  pipeline keep running instead of crashing on unseen input.

### 4. Wrap it in a `Pipeline`, exactly like Step 9

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

full_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000)),
])

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

full_pipeline.fit(X_train, y_train)
print("Test accuracy:", full_pipeline.score(X_test, y_test))

# Cross-validate the WHOLE pipeline, exactly as Step 9 did - encoding is refit on each
# training fold only, so there's no leakage from the categorical encoding either
cv_scores = cross_val_score(full_pipeline, data, target, cv=5)
print("Cross-validation accuracy:", cv_scores.mean())
```

Notice that `full_pipeline.fit(X_train, y_train)` is passed the **raw DataFrame**, city names
and all — `ColumnTransformer` handles routing each column to the right transformer internally.
This is the same leakage protection Step 9 demonstrated with `SelectKBest`: because encoding
happens inside the pipeline, `cross_val_score` refits the encoder fresh on each fold's training
data, so no information about categories in the test fold leaks into training.

### 5. Inspect what actually got built

```python
# See the expanded feature names after one-hot encoding
feature_names = full_pipeline.named_steps["preprocess"].get_feature_names_out()
print(feature_names)
```

You'll see something like `['num__age', 'num__income', 'cat__city_Berlin', 'cat__city_London',
'cat__city_Madrid', 'cat__city_Paris', 'cat__education_Bachelors', ...]` — the two numeric
columns unchanged (just scaled), and each category expanded into its own 0/1 column.

## A shortcut: selecting columns by dtype instead of by name

Naming every column by hand doesn't scale to datasets with dozens of columns. `ColumnTransformer`
supports selecting columns automatically by their pandas dtype:

```python
from sklearn.compose import make_column_selector

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), make_column_selector(dtype_include="number")),
    ("cat", OneHotEncoder(handle_unknown="ignore"), make_column_selector(dtype_include="object")),
])
```

This applies `StandardScaler` to every numeric column and `OneHotEncoder` to every
`object`-dtype (string/categorical) column, however many there are, without listing them by
name — useful once you're working with real datasets rather than a hand-built example.

## What to observe / think about

- Try removing `handle_unknown="ignore"` and manually adding a brand-new city value only to
  `X_test` (not seen in `X_train`) — you should see `OneHotEncoder` raise a `ValueError` at
  `.transform()` time, then disappear once you put `handle_unknown="ignore"` back. This is the
  exact same class of "unseen data" problem `StandardScaler` sidesteps naturally (it just
  computes a z-score for any number), but that categorical encoding does not.
- Compare `OneHotEncoder` against `OrdinalEncoder` (which maps categories to plain integers 0,
  1, 2, ... instead of separate columns) on the `education` column specifically. Since
  `education` has a natural order (`High School < Bachelors < Masters < PhD`), does encoding it
  ordinally instead of one-hot change the cross-validated accuracy? This is a good way to build
  intuition for when one-hot encoding (no assumed order) is appropriate versus ordinal encoding
  (an explicit, meaningful order).
- `ColumnTransformer` composes with everything else in the main notebook: try dropping the
  whole `full_pipeline` into `GridSearchCV` from Step 10, searching over both the
  classifier's hyperparameters (e.g. `classifier__C`) and something in the preprocessing step
  (e.g. swapping `StandardScaler` for `sklearn.preprocessing.MinMaxScaler`).
