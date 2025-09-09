from utils.db_utils import get_engine
import pandas as pd
import numpy as np
from sqlalchemy import Column, Integer, Float, Boolean, DateTime, Text, MetaData, Table


def read_data_from_db():
    engine = get_engine()
    df = pd.DataFrame()
    with engine.connect() as conn:
        result = conn.execute("SELECT * FROM property_details")
        return pd.DataFrame(result.fetchall(), columns=result.keys())


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


def fix_types(df):

    for col in [
        "postal_code",
        "number_of_bedrooms",
        "build_year",
        "number_of_facades",
        "number_of_floors",
        "number_of_bathrooms",
        "number_of_showers",
        "number_of_toilets",
        "floor_of_appartment",
        "price",
        "co2_emission",
    ]:
        convert_to_int(df, col)

    # Use extract_leading_float for all surface columns
    for col in [
        "surface_bedroom_1",
        "surface_bedroom_2",
        "surface_bedroom_3",
        "livable_surface",
        "surface_of_living_room",
        "surface_of_the_diningroom",
        "surface_kitchen",
        "surface_garden",
        "surface_terrace",
        "total_land_surface",
        "specific_primary_energy_consumption",
    ]:
        extract_leading_float(df, col)

    for col in ["validity_date_epc_peb"]:
        convert_to_date(df, col)

    for col in [
        "furnished",
        "garden",
        "terrace",
        "gas",
        "swimming_pool",
        "cellar",
        "diningroom",
        "entry_phone",
        "elevator",
        "access_for_disabled",
        "sewer_connection",
        "gas",
        "running_water",
    ]:
        convert_to_boolean(df, col)

    print(df.info())


if __name__ == "__main__":
    df = read_data_from_db()
    df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
    fix_types(df)
    # To do: remove outliers, further cleaning...

    # Insert cleaned dataframe into a new table

    engine = get_engine()
    metadata = MetaData()
    # Drop and recreate the table
    table_name = "property_details_for_analysis"
    from sqlalchemy import inspect

    # Infer SQLAlchemy column types from DataFrame dtypes
    dtype_map = {
        "int64": Integer,
        "Int64": Integer,
        "float64": Float,
        "boolean": Boolean,
        "datetime64[ns]": DateTime,
    }
    columns = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        coltype = dtype_map.get(dtype, Text)
        columns.append(Column(col, coltype))
    table = Table(table_name, metadata, *columns, extend_existing=True)
    metadata.create_all(engine)
    # Prevent NaT and NaN issues with SQLAlchemy
    records = df.replace({pd.NaT: None, np.nan: None}).to_dict(orient="records")
    # Insert data
    with engine.begin() as conn:
        conn.execute(table.insert(), records)
