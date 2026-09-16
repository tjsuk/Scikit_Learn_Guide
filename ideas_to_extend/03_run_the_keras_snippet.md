# Extending Step 12: Actually Running the Keras Snippet

## What you're extending

[Step 12](../scikit_learn_guide.ipynb) of the main notebook shows an **illustrative, not
executed** code snippet: scikit-learn handling data splitting and evaluation, with a Keras
`Sequential` model doing the actual neural network training in between. It was left unexecuted
because it needs `tensorflow` installed, which the main notebook's `requirements.txt`
deliberately doesn't include (to keep the core notebook lightweight and dependency-free). This
guide walks through actually running it, understanding every line, and comparing the result
directly against Step 11's `MLPClassifier`.

## Why this matters

Reading a code snippet and running it are different things. Running it forces you to confront
real details a snippet skips over: installation quirks, warning messages, how long training
actually takes, and — most importantly — whether the two approaches (scikit-learn's built-in
`MLPClassifier` vs. a hand-built Keras network) really do behave comparably on the same data,
or whether there are meaningful differences worth understanding.

## How to do it

### 1. Install TensorFlow

```bash
pip install tensorflow
```

On Windows, note that **native GPU support for TensorFlow was dropped after version 2.10** —
if you have an NVIDIA GPU and want it used, you'd need to install TensorFlow inside WSL2
(Windows Subsystem for Linux) instead. For this exercise, running entirely on CPU is completely
fine; the network involved is tiny (a few thousand parameters), and will train in seconds either
way. If you'd rather avoid the ambiguity entirely, `pip install tensorflow-cpu` installs an
explicitly CPU-only build.

### 2. Load the exact same data Step 11 used

Using the *same* data and the *same* split is what makes the comparison meaningful — if the
data differed, you wouldn't know whether any accuracy difference came from the model or from
the data.

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

digits = load_digits()
X_digits, y_digits = digits.data, digits.target

Xd_train, Xd_test, yd_train, yd_test = train_test_split(
    X_digits, y_digits, test_size=0.2, random_state=1, stratify=y_digits
)

scaler = StandardScaler()
Xd_train_scaled = scaler.fit_transform(Xd_train)
Xd_test_scaled = scaler.transform(Xd_test)
```

### 3. Build and train the Keras model, with every line explained

```python
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=(64,)),   # hidden layer 1: 64 neurons
    keras.layers.Dense(32, activation="relu"),                       # hidden layer 2: 32 neurons
    keras.layers.Dense(10, activation="softmax"),                    # output: 10 digit classes
])

model.compile(
    optimizer="adam",                              # how weights get updated each step
    loss="sparse_categorical_crossentropy",         # same loss Step 11's MLPClassifier minimises internally
    metrics=["accuracy"],
)

history = model.fit(
    Xd_train_scaled, yd_train,
    epochs=50,          # 50 full passes over the training data
    verbose=0,           # suppress per-epoch printouts; we'll plot the loss curve instead
    validation_split=0.1,  # hold out 10% of training data to monitor for overfitting during training
)
```

**What each piece is doing, mapped to concepts the main notebook already covered:**

- **`Dense(64, activation="relu")`** is exactly Step 11's `hidden_layer_sizes=(64, 32)` first
  layer — a fully-connected layer of 64 neurons. `MLPClassifier` uses ReLU activation by
  default too (`activation="relu"`), so this matches.
- **`softmax`** on the output layer is the same softmax explained conceptually in the
  `NumPy_Fundamentals` notebook's capstone — it turns 10 raw scores into probabilities that sum
  to 1.
- **`"adam"`** is an optimisation algorithm — a more sophisticated relative of the plain
  gradient descent used by hand in `NumPy_Fundamentals`. `MLPClassifier` also defaults to
  `solver="adam"`, so this, too, matches Step 11's setup.
- **`sparse_categorical_crossentropy`** is the same cross-entropy loss explained (and
  hand-derived!) in `NumPy_Fundamentals` Step 11c. "Sparse" just means the labels are given as
  plain integers (`3`, `7`, ...) rather than pre-converted to one-hot vectors — Keras converts
  them internally.

### 4. Evaluate and compare directly against Step 11's result

```python
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(Xd_test_scaled, verbose=0).argmax(axis=1)   # argmax: same idea as Step 9 in NumPy_Fundamentals
keras_accuracy = accuracy_score(yd_test, y_pred)

print(f"Keras Sequential accuracy:      {keras_accuracy:.4f}")
print(f"MLPClassifier accuracy (Step 11): 0.9639")   # from the main notebook's executed output
print()
print(classification_report(yd_test, y_pred))
```

### 5. Plot the loss curve, next to Step 11's

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
plt.plot(history.history["loss"], label="training loss")
plt.plot(history.history["val_loss"], label="validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Keras Sequential training loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

Compare this shape against `MLPClassifier`'s `loss_curve_` plot from Step 11 — both should show
a similar rapid initial drop followed by a long, slow flattening.

## What to observe / think about

- **The two models should land in a similar accuracy range** (typically both in the
  mid-to-high 90s% on this dataset), but rarely *identical* numbers. Differences come from: how
  weights are randomly initialised, small differences in the exact Adam implementation and
  default learning rate between libraries, and the number of training epochs vs.
  `MLPClassifier`'s `max_iter`. This is a useful, concrete lesson: "the same architecture,
  trained two different ways" is not the same as "the same model."
- **Try increasing `epochs`** from 50 to 200 and see whether Keras's accuracy catches up to, or
  overtakes, `MLPClassifier`'s. Watch the validation loss in the plot above — if it starts
  *rising* while training loss keeps falling, that's overfitting, a concept the main notebook's
  cross-validation section (Step 8) was built to help you detect and avoid.
- **Time both approaches** with `time.time()` around each `.fit()` call. For a network this
  small, `MLPClassifier` is often *faster* in wall-clock time than Keras on CPU, because Keras
  carries extra overhead (building a computation graph, etc.) that only pays off at much larger
  scale — a concrete illustration of the trade-off table in Step 11: Keras's real advantage is
  *flexibility and scale*, not raw speed on tiny problems.
- Swap `Dense(64, ...)` / `Dense(32, ...)` for different sizes, or add a third hidden layer —
  something `MLPClassifier` also supports via `hidden_layer_sizes=(64, 32, 16)` — and compare
  both libraries again. This is a good way to feel, directly, how much *more* architectural
  freedom a real deep learning framework offers even for something this simple (try, for
  example, adding `keras.layers.Dropout(0.2)` between layers — `MLPClassifier` has no
  equivalent option at all).
