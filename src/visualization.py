"""Visualisation helpers for the FraudShield project.

All functions accept a Matplotlib ``ax`` parameter so that callers can embed
plots inside larger figure layouts.  When *ax* is ``None`` a new figure is
created and ``plt.show()`` is called automatically.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def _show_or_return(ax: plt.Axes | None) -> None:
    """Call ``plt.tight_layout()`` and ``plt.show()`` when no *ax* was given."""
    if ax is None:
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# EDA plots
# ---------------------------------------------------------------------------


def plot_category_fraud_distribution(
    df: pd.DataFrame,
    ax: plt.Axes | None = None,
) -> None:
    """Bar chart – number of fraudulent transactions per category.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame containing ``category`` and ``is_fraud`` columns.
    ax : plt.Axes | None
        Optional axes to draw on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    fraud_df = df[df["is_fraud"] == 1]
    counts = fraud_df["category"].value_counts()
    sns.barplot(x=counts.index, y=counts.values, palette="viridis", ax=ax)
    ax.set_title("Distribution of Categories in Fraudulent Transactions")
    ax.set_xlabel("Category")
    ax.set_ylabel("Number of Fraudulent Transactions")
    ax.tick_params(axis="x", rotation=45)
    _show_or_return(ax)


def plot_fraud_distribution(
    df: pd.DataFrame,
    ax: plt.Axes | None = None,
) -> None:
    """Bar chart – overall fraud vs. non-fraud counts.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the ``is_fraud`` column.
    ax : plt.Axes | None
        Optional axes to draw on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    counts = df["is_fraud"].value_counts()
    colors = sns.color_palette("husl", len(counts))
    sns.barplot(x=counts.index, y=counts.values, palette=colors, ax=ax)
    ax.set_title("Distribution of Fraud")
    ax.set_xlabel("Fraud")
    ax.set_ylabel("Count")
    _show_or_return(ax)


def plot_transaction_time_distribution(
    df: pd.DataFrame,
    ax: plt.Axes | None = None,
) -> None:
    """Histogram – number of transactions by hour of day.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a ``trans_date_trans_time`` column.
    ax : plt.Axes | None
        Optional axes to draw on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    df = df.copy()
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["transaction_hour"] = df["trans_date_trans_time"].dt.hour
    sns.histplot(df["transaction_hour"], bins=24, kde=False, color="skyblue", ax=ax)
    ax.set_xlabel("Hour of the Day")
    ax.set_ylabel("Number of Transactions")
    ax.set_title("Distribution of Transaction Time")
    ax.set_xticks(range(24))
    ax.grid(axis="y")
    _show_or_return(ax)


def plot_job_sector_distribution(
    df: pd.DataFrame,
    ax: plt.Axes | None = None,
) -> None:
    """Pie chart – share of transactions per job sector.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a ``job_sector`` column.
    ax : plt.Axes | None
        Optional axes to draw on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    counts = df["job_sector"].value_counts().reset_index()
    ax.pie(
        x=counts["count"],
        labels=counts["job_sector"],
        autopct="%1.1f%%",
    )
    ax.set_title("Different Job Sectors")
    _show_or_return(ax)


def plot_job_sector_transactions(
    df: pd.DataFrame,
    ax: plt.Axes | None = None,
) -> None:
    """Bar chart – average transaction amount per job sector.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``job_sector`` and ``amt`` columns.
    ax : plt.Axes | None
        Optional axes to draw on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

    avg_amt = df.groupby("job_sector")["amt"].mean().reset_index()
    ax.bar(avg_amt["job_sector"], avg_amt["amt"], color="skyblue")
    ax.set_xlabel("Job Sector")
    ax.set_ylabel("Average Transaction Amount")
    ax.set_title("Average Transaction Amount by Job Sector")
    ax.tick_params(axis="x", rotation=90)
    _show_or_return(ax)


def plot_merchant_fraud_percentage(
    df: pd.DataFrame,
    top_n: int = 10,
    ax: plt.Axes | None = None,
) -> None:
    """Bar chart – top merchants by fraud percentage.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``merchant`` and ``is_fraud`` columns.
    top_n : int
        Number of merchants to display.
    ax : plt.Axes | None
        Optional axes to draw on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    fraud_pct = df.groupby("merchant")["is_fraud"].mean() * 100
    top = fraud_pct.nlargest(top_n)
    sns.barplot(x=top.index, y=top.values, color="darkblue", ax=ax)
    ax.set_xlabel("Merchant")
    ax.set_ylabel("Percentage of Fraudulent Transactions")
    ax.set_title(f"Top {top_n} Merchants by Percentage of Fraudulent Transactions")
    ax.tick_params(axis="x", rotation=90)
    ax.grid(axis="y")
    _show_or_return(ax)


def plot_distance_distribution(
    df: pd.DataFrame,
    ax: plt.Axes | None = None,
) -> None:
    """Histogram – distance distribution for fraudulent transactions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``distance`` and ``is_fraud`` columns.
    ax : plt.Axes | None
        Optional axes to draw on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    fraud_df = df[df["is_fraud"] == 1]
    sns.histplot(fraud_df["distance"], bins=10, kde=True, color="darkblue", ax=ax)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Number of Fraudulent Transactions")
    ax.set_title("Distribution of Distance to Fraudulent Transactions")
    ax.grid(axis="y")
    _show_or_return(ax)


def plot_transaction_frequency_fraud(
    df: pd.DataFrame,
    column: str = "transactions_last_hour",
    ax: plt.Axes | None = None,
) -> None:
    """Line chart – fraud percentage by transaction frequency bucket.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with *column* and ``is_fraud`` columns.
    column : str
        Rolling-window column name (``'transactions_last_hour'`` or
        ``'transactions_last_day'``).
    ax : plt.Axes | None
        Optional axes to draw on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

    bin_counts = df.groupby(column)["is_fraud"].agg(["count", "sum"]).reset_index()
    bin_counts.rename(
        columns={"count": "total_transactions", "sum": "fraudulent_transactions"},
        inplace=True,
    )
    bin_counts["fraud_percentage"] = (
        bin_counts["fraudulent_transactions"] / bin_counts["total_transactions"] * 100
    )
    sns.lineplot(
        x=bin_counts[column].astype(str),
        y=bin_counts["fraud_percentage"],
        marker="o",
        color="blue",
        ax=ax,
    )
    label = column.replace("_", " ").title()
    ax.set_xlabel(f"Number of Transactions ({label})")
    ax.set_ylabel("Percentage of Fraudulent Transactions (%)")
    ax.set_title(f"Fraud Percentage by {label}")
    ax.tick_params(axis="x", rotation=45)
    sns.despine(ax=ax)
    _show_or_return(ax)


def plot_correlation_heatmap(
    df: pd.DataFrame,
    ax: plt.Axes | None = None,
) -> None:
    """Heatmap – feature correlation matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Numeric DataFrame (typically the SMOTE-resampled training set).
    ax : plt.Axes | None
        Optional axes to draw on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap")
    _show_or_return(ax)


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred,
    title: str = "Confusion Matrix",
    ax: plt.Axes | None = None,
) -> None:
    """Display a confusion matrix.

    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.
    title : str
        Plot title.
    ax : plt.Axes | None
        Optional axes to draw on.
    """
    if ax is None:
        _, ax = plt.subplots()

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title(title)
    _show_or_return(ax)
