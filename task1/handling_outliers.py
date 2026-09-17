import pandas as pd
import numpy as np



def handle_outliers(doc, column, impossible_low, impossible_high, clinical_reference=None, delete_stat_outliers=False):
    """
    impossible_low / impossible_high: implausible values for THIS column, in its own unit,
                                       cross-checked against multiple health reference sites.
    clinical_reference: optional clinical threshold to compare the statistical
                         upper bound against (e.g. 126 for BLDS, 309 for tot_chole).
    delete_stat_outliers: if True, rows outside the IQR bounds are deleted
                           instead of flagged and retained.
    """
    # --- Tier 1: remove physiologically impossible values ---
    impossible_mask = (doc[column] <= impossible_low) | (doc[column] > impossible_high)
    n_impossible = impossible_mask.sum()
    print(f"{column}: {n_impossible} physiologically impossible values removed "
          f"(<= {impossible_low} or > {impossible_high})")

    doc_clean = doc[~impossible_mask].copy()

    # --- Tier 2: detect statistical outliers via log-transformed IQR ---
    log_col = np.log(doc_clean[column])
    Q1, Q3 = log_col.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_log = Q1 - 1.5 * IQR
    upper_log = Q3 + 1.5 * IQR
    lower, upper = np.exp(lower_log), np.exp(upper_log)

    print(f"{column}: IQR bounds (original scale): {lower:.2f} - {upper:.2f}")

    if clinical_reference is not None:
        if upper < clinical_reference:
            print(f"WARNING: upper bound ({upper:.2f}) is below the clinical "
                  f"reference ({clinical_reference}) for {column}.")
        else:
            print(f"Upper bound ({upper:.2f}) is above the clinical reference "
                  f"({clinical_reference}) for {column} — no conflict.")

    # --- Tier 3: either delete or flag+retain values outside the IQR bounds ---
    outlier_mask = (doc_clean[column] < lower) | (doc_clean[column] > upper)

    if delete_stat_outliers:
        n_removed = outlier_mask.sum()
        doc_clean = doc_clean[~outlier_mask].copy()
        print(f"{column}: {n_removed} statistical outlier rows deleted")
    else:
        doc_clean[f'{column}_outlier_flag'] = outlier_mask
        n_flagged = outlier_mask.sum()
        print(f"{column}: {n_flagged} rows flagged as statistical outliers (retained)")

    print(f"{column}: rows before/after: {len(doc)} / {len(doc_clean)}")

    return doc_clean



def run_all_outlier_handling():
    output_filepath = "data_cleaned_outliered.csv"
    doc = pd.read_csv("data_cleaned.csv")

    doc = handle_outliers(doc, 'BLDS', impossible_low=40, impossible_high=600, clinical_reference=126)
    doc = handle_outliers(doc, 'tot_chole', impossible_low=50, impossible_high=1000, clinical_reference=309)
    doc = handle_outliers(doc, 'triglyceride', impossible_low=10, impossible_high=2000)
    doc = handle_outliers(doc, 'gamma_GTP', impossible_low=1, impossible_high=2000)
    doc = handle_outliers(doc, 'waistline', impossible_low=40, impossible_high=200, delete_stat_outliers=True)
    doc = handle_outliers(doc, 'weight', impossible_low=20, impossible_high=250, delete_stat_outliers=True)
    doc = handle_outliers(doc, 'height', impossible_low=100, impossible_high=250, delete_stat_outliers=True)
    doc.to_csv(output_filepath, index=False)
    return doc


run_all_outlier_handling()








