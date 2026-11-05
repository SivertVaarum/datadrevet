import json, time, warnings
import pandas as pd
from scipy.stats import loguniform
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from data import load_split, make_pipeline

warnings.filterwarnings("ignore", category=ConvergenceWarning)

TUNE_ROWS = 20000      # subsample size used for tuning
N_TRIALS = 15
N_FOLDS = 3

X_train, X_test, y_train, y_test = load_split()
print(f"Train rows: {len(X_train)} | positive rate: {y_train.mean():.3f}")

# Tune on a stratified subsample of the training data only (test set untouched)
X_tune, _, y_tune, _ = train_test_split(
    X_train, y_train, train_size=TUNE_ROWS, stratify=y_train, random_state=42)

param_dist = {
    "mlp__hidden_layer_sizes": [(32,), (64,), (128,),
                                (64, 32), (128, 64), (64, 32, 16)],
    "mlp__activation": ["relu", "tanh"],
    "mlp__alpha": loguniform(1e-6, 1e-2),             # L2 penalty
    "mlp__learning_rate_init": loguniform(1e-4, 1e-2),
    "mlp__batch_size": [256, 512, 1024],
}

search = RandomizedSearchCV(
    make_pipeline(random_state=42, solver="adam", max_iter=100),
    param_distributions=param_dist,
    n_iter=N_TRIALS,
    scoring="roc_auc",
    cv=StratifiedKFold(N_FOLDS, shuffle=True, random_state=42),
    n_jobs=N_FOLDS, random_state=42, refit=False, verbose=2,
)

t0 = time.time()
search.fit(X_tune, y_tune)
tune_time = time.time() - t0

print(f"\nTuning time: {tune_time:.0f}s")
print("Best CV AUC:", round(search.best_score_, 4))
print("Best params:", search.best_params_)

best = {k.replace("mlp__", ""): v for k, v in search.best_params_.items()}
best["hidden_layer_sizes"] = list(best["hidden_layer_sizes"])   # JSON-safe
json.dump(best, open("best_params.json", "w"), indent=2)

res = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
res.to_csv("tuning_results.csv", index=False)
json.dump({"tune_time_s": tune_time, "tune_rows": TUNE_ROWS,
           "n_trials": N_TRIALS, "n_folds": N_FOLDS},
          open("tuning_meta.json", "w"), indent=2)