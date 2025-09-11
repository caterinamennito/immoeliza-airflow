from utils.db_utils import get_engine
import pandas as pd
import numpy as np
from sqlalchemy import (
    Column,
    Integer,
    Float,
    Boolean,
    DateTime,
    Text,
    MetaData,
    Table,
    UniqueConstraint,
    inspect
)
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


def prepare_data_for_analysis():
    from utils.preprocessing_utils import remove_outliers

    df = read_data_from_db()
    df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
    fix_types(df)
    # Remove outliers (analysis-relevant columns)
    df = remove_outliers(
        df,
        [
            "price",
            "livable_surface",
            "number_of_bedrooms",
            "specific_primary_energy_consumption",
        ],
    )

    # Insert cleaned dataframe into a new table

    engine = get_engine()
    metadata = MetaData()
    table_name = "property_details_for_analysis"

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

    inspector = inspect(engine)
    if inspector is not None and inspector.has_table(table_name):
        Table(table_name, metadata, autoload_with=engine).drop(engine)
    table = Table(
        table_name,
        metadata,
        *columns,
        UniqueConstraint("url", name="uq_property_details_for_analysis_url"),
        extend_existing=True
    )
    metadata.create_all(engine)
    # Prevent NaT and NaN issues with SQLAlchemy
    records = df.replace({pd.NaT: None, np.nan: None}).to_dict(orient="records")
    # Insert data
    with engine.begin() as conn:
        conn.execute(table.insert(), records)


if __name__ == "__main__":
    prepare_data_for_analysis()
