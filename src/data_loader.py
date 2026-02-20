"""Data loading utilities for the FraudShield project."""

import pandas as pd


def load_train_data(path: str = "Dataset/fraudTrain.csv") -> pd.DataFrame:
    """Load the raw training dataset.

    Parameters
    ----------
    path : str
        Path to the training CSV file.

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_csv(path)


def load_test_data(path: str = "Dataset/fraudTest.csv") -> pd.DataFrame:
    """Load the raw test dataset.

    Parameters
    ----------
    path : str
        Path to the test CSV file.

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_csv(path)


def load_clean_data(path: str = "Cleaned Data/fraudClean.csv") -> pd.DataFrame:
    """Load the pre-processed (cleaned) dataset.

    Parameters
    ----------
    path : str
        Path to the cleaned CSV file.

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_csv(path)
