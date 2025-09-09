from utils.db_utils import get_engine
import pandas as pd
import numpy as np
from sqlalchemy import Column, Integer, Float, Boolean, MetaData, Table, Text
from sqlalchemy import inspect
from utils.type_conversion import (
    convert_to_int,
    extract_leading_float,
    convert_to_date,
    convert_to_boolean,
)


def read_data_from_db():
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute("SELECT * FROM property_details")
        return pd.DataFrame(result.fetchall(), columns=result.keys())


def add_lat_lon(df, geo_csv_path):
    df_geo = pd.read_csv(geo_csv_path, sep=";")
    pc_geo_dict = {}
    for pc in df["postal_code"].astype(str).unique():
        match = df_geo.loc[df_geo["Post code"].astype(str) == pc, "Geo Point"]
        if len(match) > 0:
            pc_geo_dict[pc] = match.iloc[0]
        else:
            pc_geo_dict[pc] = None
    df["postal_code"] = df["postal_code"].astype(str)
    df["geocode"] = df["postal_code"].map(pc_geo_dict)
    df[["latitude", "longitude"]] = df["geocode"].str.split(",", expand=True)
    df["latitude"] = df["latitude"].astype(float)
    df["longitude"] = df["longitude"].astype(float)
    df.drop(columns=["geocode", "postal_code"], inplace=True)
    return df


def clean_for_regression(df):
    features = [
        "number_of_bedrooms",
        "build_year",
        "livable_surface",
        "garden",
        "terrace",
        "swimming_pool",
        "specific_primary_energy_consumption",
        "latitude",
        "longitude",
    ]

    geo_csv_path = "scraper/src/georef-belgium-postal-codes@public.csv"
    # Add latitude and longitude columns based on postal_code (must be present)
    df = add_lat_lon(df, geo_csv_path)

    target = "price"
    # Keep only features + target
    keep_cols = features + [target]
    df = df[keep_cols]

    # Drop rows with missing target or features
    df = df.dropna(subset=[target] + features)

    for col in [
        "number_of_bedrooms",
        "build_year",
        "price",
    ]:
        convert_to_int(df, col)

    # Use extract_leading_float for all surface columns
    for col in [
        "livable_surface",
        "specific_primary_energy_consumption",
    ]:
        extract_leading_float(df, col)

    for col in [
        "garden",
        "terrace",
        "swimming_pool",
    ]:
        convert_to_boolean(df, col)

    # hot-encode booleans
    bool_cols = ["garden", "terrace", "swimming_pool"]
    for col in bool_cols:
        df[col] = df[col].astype(int)

    # Final check: ensure all columns are numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()
    return df


if __name__ == "__main__":
    df = read_data_from_db()
    df_clean = clean_for_regression(df)
    print(df_clean.info())

    # Save to new table
    engine = get_engine()
    metadata = MetaData()
    table_name = "property_details_for_regression"
    inspector = inspect(engine)
    if inspector.has_table(table_name):
        Table(table_name, metadata, autoload_with=engine).drop(engine)
    # Infer types
    dtype_map = {
        "int64": Integer,
        "Int64": Integer,
        "float64": Float,
    }
    columns = []
    for col in df_clean.columns:
        dtype = str(df_clean[col].dtype)
        coltype = dtype_map.get(dtype, Float)
        columns.append(Column(col, coltype))
    table = Table(table_name, metadata, *columns, extend_existing=True)
    metadata.create_all(engine)
    records = df_clean.replace({pd.NaT: None, np.nan: None}).to_dict(orient="records")
    with engine.begin() as conn:
        conn.execute(table.insert(), records)
