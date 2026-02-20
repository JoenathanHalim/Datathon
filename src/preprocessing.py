"""Preprocessing utilities for the FraudShield project.

Functions
---------
encode_categorical_features : Label-encode specified columns in-place.
apply_smote                 : Oversample the minority class with SMOTE.
scale_features              : Standardise feature matrices.
apply_isolation_forest      : Append anomaly scores from IsolationForest.
prepare_model_data          : Full preprocessing pipeline for modelling.
"""

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def encode_categorical_features(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Label-encode categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list[str] | None
        Column names to encode.  Defaults to ``['merchant', 'category', 'job']``.

    Returns
    -------
    pd.DataFrame
        DataFrame with encoded columns (copy).
    """
    if columns is None:
        columns = ["merchant", "category", "job"]
    df = df.copy()
    le = LabelEncoder()
    for col in columns:
        df[col] = le.fit_transform(df[col])
    return df


def apply_smote(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Oversample the minority class using SMOTE.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Resampled ``(X, y)``.
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standardise feature matrices using :class:`~sklearn.preprocessing.StandardScaler`.

    The scaler is fitted on *X_train* only and applied to both splits.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    X_test : pd.DataFrame
        Test feature matrix.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Scaled ``(X_train, X_test)``.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def apply_isolation_forest(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: list[str],
    contamination: float = 0.02,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Append an ``isolation_forest`` anomaly-score column to train/test splits.

    Parameters
    ----------
    X_train : array-like
        Training feature matrix (scaled).
    X_test : array-like
        Test feature matrix (scaled).
    feature_names : list[str]
        Original column names used to reconstruct DataFrames.
    contamination : float
        Expected proportion of outliers in the dataset.
    random_state : int
        Random seed.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(X_train_combined, X_test_combined)`` with the extra column.
    """
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    iso_forest.fit(X_train)

    X_train_combined = pd.DataFrame(X_train, columns=feature_names)
    X_train_combined["isolation_forest"] = iso_forest.predict(X_train)
    X_train_combined["isolation_forest"] = X_train_combined["isolation_forest"].map(
        {1: 0, -1: 1}
    )

    X_test_combined = pd.DataFrame(X_test, columns=feature_names)
    X_test_combined["isolation_forest"] = iso_forest.predict(X_test)
    X_test_combined["isolation_forest"] = X_test_combined["isolation_forest"].map(
        {1: 0, -1: 1}
    )

    return X_train_combined, X_test_combined


def prepare_model_data(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str = "is_fraud",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """Full preprocessing pipeline for modelling.

    Steps
    -----
    1. Select *feature_cols* and *target_col* from *df*.
    2. Train / test split (stratified).
    3. Apply SMOTE **only** to the training set (avoids data leakage).
    4. Standardise features (scaler fitted on training set only).
    5. Append IsolationForest anomaly feature to train and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned input DataFrame.
    feature_cols : list[str] | None
        Columns to use as features.  Defaults to the numeric model features.
    target_col : str
        Name of the target column.
    test_size : float
        Fraction of data reserved for testing.
    random_state : int
        Random seed used throughout.

    Returns
    -------
    tuple
        ``(X_train_combined, X_test_combined, Y_train, Y_test)``
    """
    if feature_cols is None:
        feature_cols = [
            "merchant",
            "category",
            "amt",
            "job",
            "age_at_transaction",
            "distance",
            "transactions_last_hour",
            "transactions_last_day",
        ]

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    X_train_resampled, Y_train = apply_smote(X_train, Y_train, random_state=random_state)

    X_train_scaled, X_test_scaled = scale_features(X_train_resampled, X_test)

    X_train_combined, X_test_combined = apply_isolation_forest(
        X_train_scaled, X_test_scaled, feature_names=feature_cols
    )

    return X_train_combined, X_test_combined, Y_train, Y_test
