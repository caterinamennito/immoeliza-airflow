def remove_outliers(df, cols, method="iqr", factor=1.5):
    """
    Remove outliers from DataFrame for specified columns using IQR or z-score method.
    Args:
        df: DataFrame
        cols: list of column names to check for outliers
        method: 'iqr' (default) or 'zscore'
        factor: IQR multiplier (default 1.5) or z-score threshold (default 3)
    Returns:
        DataFrame with outliers removed
    """
    df_clean = df.copy()
    if method == "iqr":
        for col in cols:
            if col in df_clean.columns:
                q1 = df_clean[col].quantile(0.25)
                q3 = df_clean[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - factor * iqr
                upper = q3 + factor * iqr
                df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    elif method == "zscore":
        from scipy.stats import zscore

        for col in cols:
            if col in df_clean.columns:
                z = np.abs(zscore(df_clean[col].dropna()))
                mask = z < factor
                df_clean = df_clean.loc[df_clean[col].dropna().index[mask]]
    return df_clean


import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def compute_epc_score(df, col="specific_primary_energy_consumption"):
    """
    Compute EPC score from specific_primary_energy_consumption using standard bins/labels.
    Adds 'epc_score' column to df.
    """
    bins = [0, 100, 200, 300, 400, 500, float("inf")]
    labels = ["A", "B", "C", "D", "E", "F"]
    df["epc_score"] = pd.cut(df[col], bins=bins, labels=labels).fillna("E")
    return df


def ordinal_encode_epc_score(df, col="epc_score"):
    """
    Ordinal encode the EPC score column (A-F) into epc_score_encoded (0-5).
    """
    oe = OrdinalEncoder(categories=[["A", "B", "C", "D", "E", "F"]])
    df["epc_score_encoded"] = oe.fit_transform(df[[col]])
    return df


def ordinal_encode_state(df, col="state_of_the_property"):
    """
    Ordinal encode the state_of_the_property column into state_of_property_encoded.
    """
    oe = OrdinalEncoder(
        categories=[
            [
                "To renovate",
                "To demolish",
                "New",
                "Under construction",
                "To be renovated",
                "Normal",
                "To restore",
                "Fully renovated",
                "Excellent",
            ]
        ]
    )
    df[[col]] = df[[col]].fillna("Normal")
    df["state_of_property_encoded"] = oe.fit_transform(df[[col]])
    return df
