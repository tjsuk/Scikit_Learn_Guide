# Extending Step 11: Rerunning the Capstone on Full-Resolution MNIST

## What you're extending

[Step 11](../scikit_learn_guide.ipynb)'s capstone trained several classifiers, including
`MLPClassifier`, on scikit-learn's built-in `load_digits()` dataset: 1,797 images, each an 8×8
grid of pixels (64 features total). This is a deliberately small, low-resolution dataset chosen
so every cell in the main notebook runs in well under a second. The famous **MNIST** dataset is
the same *idea* — handwritten digits, 0 through 9 — at real scale: 70,000 images, each 28×28
pixels (784 features), the standard benchmark dataset most people mean when they say "the digits
dataset" in a machine learning context.

## Why this matters

Small examples are great for learning *because* they run instantly, but that speed can hide how
differently your tools behave once data actually gets large. Rerunning the exact same code from
Step 11 on data roughly **39× larger** (`70,000 / 1,797 ≈ 39×` more images, `784 / 64 = 12.25×`
more features per image) is one of the best ways to build real intuition for *why* the "scale"
row in Step 11's scikit-learn-vs-deep-learning comparison table matters in practice, rather than
just reading it as a claim.

## How to do it

### 1. Fetch the full dataset

```python
from sklearn.datasets import fetch_openml
import time

start = time.time()
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
print(f"Download/load took {time.time() - start:.1f}s")

X_mnist, y_mnist = mnist.data, mnist.target.astype(int)
print("X_mnist shape:", X_mnist.shape)   # (70000, 784)
print("y_mnist shape:", y_mnist.shape)   # (70000,)
```

The first call downloads the dataset (roughly 50MB) from OpenML and caches it locally —
subsequent calls load instantly from that cache instead of re-downloading. If you're on a
restricted or offline network, be aware this step needs internet access the first time only.

### 2. Use MNIST's conventional train/test split

Unlike Step 11, which used `train_test_split` with a random split, MNIST has a **standard,
universally-used split**: the first 60,000 images for training, the last 10,000 for testing.
Using this exact split is what makes your results comparable to published MNIST benchmarks
elsewhere.

```python
X_train, X_test = X_mnist[:60000], X_mnist[60000:]
y_train, y_test = y_mnist[:60000], y_mnist[60000:]

print("Training images:", X_train.shape[0])
print("Test images:    ", X_test.shape[0])
```

### 3. Normalise pixel values — note the different scale

MNIST pixel values range **0-255** (standard 8-bit grayscale), unlike `load_digits()`'s
0-16 range that Step 11 divided by. Adjust accordingly:

```python
X_train_scaled = X_train / 255.0
X_test_scaled = X_test / 255.0
```

(`StandardScaler` would also work here, exactly as Step 11 used it in a `Pipeline` — dividing
by the fixed maximum is a simpler alternative that works well because every pixel already
shares the same natural 0-255 range, unlike Step 6's income/experience example where different
features had *different* natural units.)

### 4. Rerun Step 11's model comparison — but budget your time first

**Do not casually copy Step 11's full comparison loop unmodified.** On 60,000 training images
of 784 features, `RandomForestClassifier(n_estimators=200)` and `SVC()` can each take many
minutes; `MLPClassifier` can take longer still. Time a single model first, on a **subset**, to
estimate the real cost before committing to a full run:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Time ONE model on a small subset first
subset_size = 5000
start = time.time()
quick_model = LogisticRegression(max_iter=1000)
quick_model.fit(X_train_scaled[:subset_size], y_train[:subset_size])
subset_time = time.time() - start
print(f"Logistic Regression on {subset_size} samples took {subset_time:.1f}s")
print(f"Rough estimate for all 60,000: {subset_time * (60000 / subset_size):.0f}s")
```

Once you have a time budget in mind, run the full comparison, reusing Step 11's exact
structure:

```python
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Neural Network (MLPClassifier)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=50, random_state=42),
}

results = {}
for name, model in models.items():
    start = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start
    accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
    results[name] = accuracy
    print(f"{name:<32} accuracy={accuracy:.4f}   train_time={train_time:.1f}s")
```

(Notice `MLPClassifier`'s `max_iter` was reduced from Step 11's `2000` to `50` here — left at
`2000`, it would take dramatically longer on this much more data. This is itself part of the
lesson: at real scale, you can no longer just leave every setting at a small-dataset default.)

## What to observe / think about

- **Compare training times directly against Step 11's runtimes on the tiny `load_digits()`
  data** (Random Forest: 0.31s, MLPClassifier: 0.45s, both on 1,437 training images). Work out
  the actual slowdown factor for each model — is it proportional to the ~39× increase in
  images, worse, or better? (Different algorithms scale with data size very differently: a
  linear model like `LogisticRegression` tends to scale close to linearly with sample count,
  while some tree-based and kernel methods scale considerably worse.)
- **Compare final accuracy too.** Full MNIST is generally *easier* to get high accuracy on than
  you might expect, purely because there's so much more training data for the model to learn
  from — even a simple model can end up performing better here than a more complex model did on
  Step 11's smaller dataset. This is a good concrete example of the general principle "more
  data often beats a fancier algorithm."
- **This is exactly the scale where Step 11's comparison table's claims start to bite.**
  Rerun just the `MLPClassifier` portion, and separately (following
  [`03_run_the_keras_snippet.md`](03_run_the_keras_snippet.md)) train an equivalent Keras
  network on the same MNIST split. If you have access to a GPU, this is where you'd finally see
  a dramatic training-time advantage for Keras/PyTorch over scikit-learn's CPU-only
  `MLPClassifier` — a difference that was completely invisible on the tiny 1,797-image dataset,
  because everything finished in under a second regardless.
- Try reducing training set size deliberately (e.g. `X_train_scaled[:2000]`) and watch how
  quickly each model's accuracy degrades — this builds intuition for **learning curves**: how
  much a model's performance depends on how much data it got to train on, a question
  scikit-learn has a dedicated tool for — look up `sklearn.model_selection.learning_curve` to
  plot this systematically instead of checking a few sizes by hand.
