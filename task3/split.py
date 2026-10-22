from sklearn.model_selection import train_test_split

from config import RANDOM_STATE, TARGET, TEST_SIZE


def split_train_test(df, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Stratified hold-out split so both sets keep the ~7.7% diabetes prevalence."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def stratified_subsample(X, y, n_samples, random_state=RANDOM_STATE):
    """Smaller class-balanced-as-original sample for models that scale poorly with n
    (kernel SVM is O(n^2)-O(n^3), KNN prediction is O(n) per query)."""
    X_sample, _, y_sample, _ = train_test_split(
        X, y, train_size=n_samples, stratify=y, random_state=random_state
    )
    return X_sample, y_sample
