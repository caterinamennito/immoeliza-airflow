import pandas as pd


def extract_leading_float(df, col):
    df[col] = (
        df[col]
        .astype(str)
        .str.extract(r"([\d.,]+)", expand=False)
        .str.replace(",", ".")
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")


def convert_to_int(df, col):
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")


def convert_to_date(df, col):
    df[col] = pd.to_datetime(df[col], errors="coerce").dt.floor("D")
    # Replace NaT with None for SQL compatibility
    df[col] = df[col].where(df[col].notna(), None)


def convert_to_boolean(df, col):
    df[col] = df[col].map({"Yes": True, "No": False}).fillna(False).astype("boolean")

