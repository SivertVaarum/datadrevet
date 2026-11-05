import json, time, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                             f1_score, accuracy_score, confusion_matrix,
                             precision_recall_curve)
from sklearn.model_selection import train_test_split
from data import load_split, make_pipeline

warnings.filterwarnings("ignore", category=ConvergenceWarning)

N_SEEDS = 5          # start with 1; raise to 3 once you know the fit time
FIT_ROWS = None      # e.g. 300000 to train on a subsample; None = all training rows
MAX_ITER = 50

X_train, X_test, y_train, y_test = load_split()

params = json.load(open("best_params.json"))
params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])
# Tuning used 20k rows; on 900k rows small batches are very slow
params["batch_size"] = max(params.get("batch_size", 256), 512)

if FIT_ROWS and FIT_ROWS < len(X_train):
    X_train, _, y_train, _ = train_test_split(
        X_train, y_train, train_size=FIT_ROWS, stratify=y_train, random_state=0)

# Hold out part of train to choose the decision threshold (never the test set)
X_fit, X_thr, y_fit, y_thr = train_test_split(
    X_train, y_train, test_size=0.15, stratify=y_train, random_state=0)
print(f"Fit rows: {len(X_fit)} | threshold rows: {len(X_thr)} | test rows: {len(X_test)}")

rows = []
for seed in range(N_SEEDS):
    pipe = make_pipeline(random_state=seed, solver="adam",
                         max_iter=MAX_ITER, **params)

    t0 = time.time()
    pipe.fit(X_fit, y_fit)
    fit_time = time.time() - t0

    # Threshold maximizing F1 on the held-out slice of train
    p_thr = pipe.predict_proba(X_thr)[:, 1]
    prec, rec, thr = precision_recall_curve(y_thr, p_thr)
    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    threshold = float(thr[np.argmax(f1s)])

    probs = pipe.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    rows.append(dict(
        seed=seed, threshold=threshold, fit_time=fit_time,
        auc=roc_auc_score(y_test, probs), acc=accuracy_score(y_test, preds),
        precision=precision_score(y_test, preds, zero_division=0),
        recall=recall_score(y_test, preds), f1=f1_score(y_test, preds)))
    last_pipe, last_preds = pipe, preds
    print(f"seed {seed} done in {fit_time:.0f}s | epochs run: {pipe.named_steps['mlp'].n_iter_}")

mlp = last_pipe.named_steps["mlp"]
n_params = int(sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_))

print("\nParams used:", params)
print("Parameter count:", n_params)
print(f"\nTest results over {N_SEEDS} seed(s) (mean ± std):")
summary = {}
for k in ["auc", "acc", "precision", "recall", "f1", "fit_time"]:
    v = [r[k] for r in rows]
    summary[k] = (float(np.mean(v)), float(np.std(v)))
    print(f"  {k:10s} {np.mean(v):.4f} ± {np.std(v):.4f}")
print("\nConfusion matrix (last seed):\n", confusion_matrix(y_test, last_preds))

json.dump({"params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()},
           "n_params": n_params, "summary": summary, "runs": rows},
          open("final_results.json", "w"), indent=2)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].plot(mlp.loss_curve_); ax[0].set(title="Training loss", xlabel="epoch")
ax[1].plot(mlp.validation_scores_); ax[1].set(title="Validation accuracy", xlabel="epoch")
plt.tight_layout(); plt.savefig("learning_curves.png", dpi=150)