"""Feature engineering for the FraudShield project.

Functions
---------
assign_sector               : Map a job title to a high-level industry sector.
compute_age_at_transaction  : Add an ``age_at_transaction`` column.
compute_distance            : Add a Haversine ``distance`` column (km).
compute_transaction_frequency : Add rolling transaction-count windows.
engineer_features           : Apply all feature-engineering steps in one call.
"""

from math import atan2, cos, radians, sin, sqrt

import pandas as pd

# ---------------------------------------------------------------------------
# Sector keyword lookup table
# ---------------------------------------------------------------------------

SECTOR_BAG: dict[str, list[str]] = {
    "IT": [
        "engineer", "developer", "programmer", "software", "IT", "technician",
        "architect", "system", "network", "administrator", "data scientist",
        "cybersecurity", "web developer", "analyst", "database", "devops",
    ],
    "Education": [
        "teacher", "professor", "educator", "trainer", "lecturer", "scientist",
        "Orthoptist", "tutor", "principal", "instructor", "counselor",
        "academic", "researcher", "dean", "school", "headmaster",
    ],
    "Healthcare": [
        "doctor", "nurse", "medical", "therapist", "pharmacist", "health",
        "surgeon", "dentist", "clinician", "physician", "optometrist",
        "radiologist", "paramedic", "midwife", "veterinarian", "psychiatrist",
    ],
    "Finance": [
        "analyst", "accountant", "auditor", "banker", "financial", "investment",
        "controller", "broker", "consultant", "treasurer", "loan officer",
        "trader", "actuary", "economist", "portfolio", "credit",
    ],
    "Marketing": [
        "manager", "executive", "specialist", "consultant", "advertising",
        "public relations", "strategist", "director", "coordinator", "brand",
        "SEO", "content", "digital", "market research", "social media",
        "copywriter",
    ],
    "Manufacturing": [
        "operator", "mechanic", "assembler", "fabricator", "engineer",
        "technician", "welder", "planner", "quality", "machinist",
        "production", "inspector", "supervisor", "foreman", "toolmaker", "CNC",
    ],
    "Retail": [
        "cashier", "salesperson", "store", "associate", "manager", "clerk",
        "shopkeeper", "merchandiser", "assistant", "retail", "customer service",
        "sales", "inventory", "buyer", "stocker", "checkout",
    ],
    "Legal": [
        "lawyer", "attorney", "paralegal", "judge", "legal", "solicitor",
        "notary", "clerk", "litigator", "advocate", "barrister", "counsel",
        "magistrate", "prosecutor", "defense", "compliance",
    ],
    "Hospitality": [
        "chef", "waiter", "bartender", "host", "manager", "receptionist",
        "housekeeper", "concierge", "caterer", "cook", "hotel", "tour guide",
        "event planner", "sous chef", "sommelier", "valet",
    ],
    "Construction": [
        "builder", "carpenter", "electrician", "plumber", "architect",
        "project manager", "site manager", "surveyor", "foreman", "bricklayer",
        "roofer", "civil engineer", "construction", "contractor", "inspector",
        "draftsman",
    ],
}


# ---------------------------------------------------------------------------
# Individual feature helpers
# ---------------------------------------------------------------------------


def assign_sector(job_title: str) -> str:
    """Return the industry sector for a given job title string.

    Parameters
    ----------
    job_title : str
        Raw job title from the dataset.

    Returns
    -------
    str
        One of the keys in :data:`SECTOR_BAG`, or ``"Other"``.
    """
    for sector, keywords in SECTOR_BAG.items():
        for keyword in keywords:
            if keyword in job_title:
                return sector
    return "Other"


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points (km).

    Parameters
    ----------
    lat1, lon1 : float
        Latitude and longitude of the first point (degrees).
    lat2, lon2 : float
        Latitude and longitude of the second point (degrees).

    Returns
    -------
    float
        Distance in kilometres.
    """
    R = 6371.0  # Earth radius in km
    lat1_r, lon1_r = radians(lat1), radians(lon1)
    lat2_r, lon2_r = radians(lat2), radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = sin(dlat / 2) ** 2 + cos(lat1_r) * cos(lat2_r) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def compute_age_at_transaction(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``age_at_transaction`` column to *df* (modifies a copy).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``trans_date_trans_time`` and ``dob`` columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with an additional ``age_at_transaction`` column.
    """
    df = df.copy()
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"])
    df["age_at_transaction"] = df.apply(
        lambda row: row["trans_date_trans_time"].year - row["dob"].year
        - (
            (row["trans_date_trans_time"].month, row["trans_date_trans_time"].day)
            < (row["dob"].month, row["dob"].day)
        ),
        axis=1,
    )
    return df


def compute_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``distance`` column (Haversine km) to *df* (modifies a copy).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``lat``, ``long``, ``merch_lat``, ``merch_long`` columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with an additional ``distance`` column.
    """
    df = df.copy()
    df["distance"] = df.apply(
        lambda row: haversine_distance(
            row["lat"], row["long"], row["merch_lat"], row["merch_long"]
        ),
        axis=1,
    )
    return df


def compute_transaction_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling transaction-count columns to *df* (modifies a copy).

    Adds ``transactions_last_hour`` and ``transactions_last_day`` by grouping
    on ``cc_num`` and applying a time-based rolling window on ``amt``.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``cc_num``, ``trans_date_trans_time``, and ``amt``.

    Returns
    -------
    pd.DataFrame
        DataFrame with two additional frequency columns.
    """
    df = df.copy()
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df = df.sort_values(by=["cc_num", "trans_date_trans_time"])
    df = df.set_index("trans_date_trans_time")
    df["transactions_last_hour"] = (
        df.groupby("cc_num")["amt"].rolling("1h").count().reset_index(0, drop=True)
    )
    df["transactions_last_day"] = (
        df.groupby("cc_num")["amt"].rolling("1D").count().reset_index(0, drop=True)
    )
    df = df.reset_index()
    return df


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature-engineering steps to *df*.

    Steps applied (in order):

    1. :func:`compute_age_at_transaction`
    2. :func:`compute_distance`
    3. :func:`compute_transaction_frequency`
    4. Add ``job_sector`` via :func:`assign_sector`

    Parameters
    ----------
    df : pd.DataFrame
        Raw training or test DataFrame.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame.
    """
    df = compute_age_at_transaction(df)
    df = compute_distance(df)
    df = compute_transaction_frequency(df)
    df["job_sector"] = df["job"].apply(assign_sector)
    return df
