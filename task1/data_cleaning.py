import pandas as pd

doc = pd.read_csv("smoking_driking_dataset_Ver01.csv")


def encode_sex_column(doc):
    unknown_rows = ~doc['sex'].isin(['Male', 'Female'])
    num_unknown_rows = unknown_rows.sum()
    print(f"Number of rows with unknown/other value in 'sex': {num_unknown_rows}")

    doc['sex'] = doc['sex'].map({'Male': 0, 'Female': 1}).fillna(10).astype(int)

    doc.to_csv("smoking_driking_dataset_encoded.csv", index=False)
    return doc


encode_sex_column(doc)

