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
