# Scikit-learn: From Raw Data to Trained Models

A self-contained Jupyter notebook that teaches **scikit-learn** — Python's core library for
classical machine learning — from first principles up to an undergraduate ("degree level 5")
depth. It's written as a **learning exercise**, not just a reference: every step includes
plain-English explanations of *what* the code does and *why*, and backs up each claim with a
worked, numeric demonstration rather than just asserting it.

It starts by writing a classifier by hand to feel the problem scikit-learn solves, then builds
up through the Estimator API, preprocessing, evaluation, pipelines, and hyperparameter search,
finishing with a capstone that trains a real neural network (`MLPClassifier`) on the same
handwritten digit images used in the companion **NumPy_Fundamentals** notebook — so the two
notebooks can be read together as "build it by hand, then see the toolkit that does it for you."

## What's inside

1. **Import scikit-learn** and see what it gives you
2. **The problem scikit-learn solves** — implementing k-nearest neighbors by hand, then
   confirming it matches `KNeighborsClassifier` exactly
3. **The core API** — Estimators, Transformers, Predictors (`fit`/`predict`/`transform`)
4. **Loading and splitting data** — the Iris dataset, `train_test_split`
5. **Why the API matters** — five different algorithms, trained and scored with identical code
6. **Why preprocessing matters** — a deliberately extreme demo where feature scaling alone
   turns a 43%-accurate model into a 99%-accurate one
7. **Training and evaluating a classifier** — `classification_report`, confusion matrices
8. **Why one train/test split isn't enough** — measuring split-to-split variance directly,
   then fixing it with `cross_val_score`
9. **Pipelines and data leakage** — a leaky feature-selection pipeline claims ~70% accuracy on
   pure random noise; the correct pipeline correctly reports ~50% (chance)
10. **Hyperparameter tuning** — a hand-rolled grid search vs. `GridSearchCV`, confirmed to
    agree on the same best answer
11. **Capstone**: training `MLPClassifier` (scikit-learn's neural network) on 1,797 real
    handwritten digit images, benchmarked against logistic regression, a random forest, and an
    SVM — with a direct comparison to the from-scratch NumPy network's accuracy
12. **How scikit-learn fits alongside deep learning frameworks** — TensorFlow/Keras and
    PyTorch, with an illustrative (non-executed) Keras snippet

It finishes with a **Summary** recapping the whole notebook and a glossary of key terms
(Estimator, Transformer, Pipeline, cross-validation, data leakage, hyperparameter, MLP), plus
**Ideas to extend** — `RandomizedSearchCV`, `ColumnTransformer`, SciKeras, model persistence
with `joblib`, and scaling up to full MNIST.

## Requirements

- **Python 3.9+**
- pip

## Setup

1. **Clone or download this repository.**

2. **Create a virtual environment** in the project folder:

   ```bash
   python -m venv .venv
   ```

3. **Activate it.**

   Windows (PowerShell/cmd):
   ```bash
   .venv\Scripts\activate
   ```

   macOS/Linux:
   ```bash
   source .venv/bin/activate
   ```

4. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

5. **Register the environment as a Jupyter kernel** (so the notebook can find your installed
   packages):

   ```bash
   python -m ipykernel install --user --name=scikit-learn-guide-venv --display-name "Python (scikit-learn-guide-venv)"
   ```

## Running the notebook

Launch Jupyter Lab **using the venv's own executable** rather than relying on `activate` alone
(if another Python install is earlier on your `PATH`, a plain `jupyter lab` command can silently
launch the wrong environment):

```bash
.venv\Scripts\jupyter-lab.exe
```

(macOS/Linux: `.venv/bin/jupyter-lab`, after activating the venv.)

Open `scikit_learn_guide.ipynb`, and make sure it's using the **"Python
(scikit-learn-guide-venv)"** kernel (in VS Code: click the kernel picker in the top-right of the
notebook; in Jupyter Lab: use the **Kernel → Change Kernel** menu).

Then run the cells from top to bottom, reading the explanations as you go. The whole notebook
(including training the capstone neural network) runs in well under a minute on a typical CPU.
Every dataset used (Iris, the synthetic scaling/leakage demos, and the handwritten digits) is
either built into scikit-learn or generated on the fly, so nothing needs to be downloaded — the
notebook works fully offline.

## Notes

- The capstone (Step 11) deliberately reuses the same 8x8 handwritten digits dataset as the
  `NumPy_Fundamentals` notebook's capstone, so the two can be compared directly: a from-scratch
  NumPy network there, and several one-line scikit-learn models here.
- Step 9's data-leakage demo uses 500 purely random features and purely random labels on
  purpose — the "correct" accuracy really should sit around 50%, and the notebook shows exactly
  how easy it is to accidentally report something far higher.

## Ideas to extend

- Use `RandomizedSearchCV` on a much larger hyperparameter grid and compare how close it gets
  to the true best score in a fraction of the time
- Try `ColumnTransformer` on a dataset with mixed numeric and categorical features
- Install `tensorflow` and run Step 12's illustrative Keras snippet for real, then compare its
  accuracy against Step 11's `MLPClassifier`
- Look up **SciKeras**, which lets a Keras model be tuned with `GridSearchCV` just like
  `RandomForestClassifier` was in Step 10
- Save a trained pipeline with `joblib.dump`/`joblib.load` and reload it in a fresh session
- Rerun the capstone on full-resolution MNIST (`sklearn.datasets.fetch_openml('mnist_784')`)
  and compare training time and accuracy against the smaller digits dataset used here

Each of these has a detailed, step-by-step walkthrough with full explanations and runnable code
in [`ideas_to_extend/`](ideas_to_extend/README.md).
