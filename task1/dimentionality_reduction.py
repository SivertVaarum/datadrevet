import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# --- Update this to your actual cleaned/scaled CSV file ---
INPUT_FILE = "data_transformed.csv"

doc = pd.read_csv(INPUT_FILE)

# Columns used for PCA: continuous predictors only.
# Excluded: BLDS (used to derive the diabetes target -> leakage),
# categorical/ordinal columns (hear_left/right, urine_protein,
# SMK_stat_type_cd, DRINK, sex_female), and the outlier flag columns
# (boolean indicators, not continuous measurements).
pca_columns = [
    'age', 'height', 'weight', 'waistline',
    'sight_left', 'sight_right',
    'SBP', 'DBP',
    'tot_chole', 'HDL_chole', 'LDL_chole', 'triglyceride',
    'hemoglobin', 'serum_creatinine',
    'SGOT_AST', 'SGOT_ALT', 'gamma_GTP'
]

missing = [c for c in pca_columns if c not in doc.columns]
if missing:
    raise ValueError(f"These expected columns are missing from the file: {missing}")

X = doc[pca_columns].copy()

# Sanity check: confirm the data is already standardized (mean ~0, std ~1).
# PCA assumes standardized input; if this check fails, the columns were
# not scaled as expected and should be standardized before proceeding.
means = X.mean()
stds = X.std()
print("Sanity check (should be close to 0 and 1 if already standardized):")
print(f"Mean range: {means.min():.3f} to {means.max():.3f}")
print(f"Std range: {stds.min():.3f} to {stds.max():.3f}")
print()

# --- Run PCA ---
pca = PCA()
principal_components = pca.fit_transform(X)

explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = explained_variance_ratio.cumsum()

print(f"Number of original features: {len(pca_columns)}")
print()
print("Explained variance ratio per component:")
for i, (var, cum) in enumerate(zip(explained_variance_ratio, cumulative_variance), start=1):
    print(f"  PC{i}: {var:.4f}  (cumulative: {cum:.4f})")

# Number of components needed to reach common variance thresholds
for threshold in [0.90, 0.95]:
    n_components = np.argmax(cumulative_variance >= threshold) + 1
    print(f"\nComponents needed for {threshold*100:.0f}% variance: {n_components} "
          f"(out of {len(pca_columns)} original features)")

# --- Scree plot ---
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.axhline(y=0.90, color='r', linestyle='--', label='90% variance')
plt.axhline(y=0.95, color='g', linestyle='--', label='95% variance')
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA - Cumulative Explained Variance")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pca_scree_plot.png", dpi=150)
print("\nScree plot saved to pca_scree_plot.png")

# --- Save transformed components (optional, for inspection) ---
pc_columns = [f"PC{i+1}" for i in range(principal_components.shape[1])]
pca_df = pd.DataFrame(principal_components, columns=pc_columns)
pca_df.to_csv("pca_transformed_components.csv", index=False)
print("Transformed components saved to pca_transformed_components.csv")