import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
from regression_model.preprocessing import clean_for_regression
from utils.preprocessing_utils import (
    compute_epc_score,
    ordinal_encode_epc_score,
    ordinal_encode_state,
)


def predict_price(df):
    with open("/opt/airflow/regression_model/model.pkl", "rb") as f:
        model = pickle.load(f)
        log_pred = model.predict(df)[0]
        return np.expm1(log_pred)


def postal_code_to_latlon(
    postal_code, geo_csv_path="scraper/src/georef-belgium-postal-codes@public.csv"
):
    df_geo = pd.read_csv(geo_csv_path, sep=";")
    match = df_geo.loc[df_geo["Post code"].astype(str) == str(postal_code), "Geo Point"]
    if len(match) > 0:
        lat, lon = match.iloc[0].split(",")
        return float(lat), float(lon)
    else:
        return np.nan, np.nan


st.set_page_config(layout="centered")
st.title("Property Price Estimator")

# Show last updated at (latest scrape_timestamp from property_details)
import os

AIRFLOW_CONN = os.environ.get(
    "AIRFLOW__CORE__SQL_ALCHEMY_CONN",
    "postgresql+psycopg2://airflow:airflow@postgres:5432/airflow",
)
try:
    import pytz
    from datetime import datetime

    engine = create_engine(AIRFLOW_CONN)
    last_update = pd.read_sql(
        "SELECT MAX(scrape_timestamp) as last_updated FROM property_details",
        engine,
    )
    last_updated = last_update["last_updated"].iloc[0]
    if pd.notnull(last_updated):
        # Convert to CEST (Europe/Brussels)
        if not pd.isna(last_updated):
            if not hasattr(last_updated, "tzinfo") or last_updated.tzinfo is None:
                last_updated = pd.Timestamp(last_updated).tz_localize("UTC")
            last_updated_cest = last_updated.tz_convert("Europe/Brussels")
            st.info(
                f"Last updated at: {last_updated_cest.strftime('%Y-%m-%d %H:%M:%S %Z')}"
            )
        else:
            st.info("Last updated at: No data available.")
    else:
        st.info("Last updated at: No data available.")
except Exception as e:
    st.info(f"Last updated at: (error loading timestamp: {e})")

tab1, tab2 = st.tabs(["Predict Price", "Data Dashboard"])

with tab1:
    with st.form("prediction_form"):
        number_of_bedrooms = st.number_input(
            "Number of bedrooms", min_value=0, max_value=20, value=2
        )
        livable_surface = st.number_input(
            "Livable surface (m²)", min_value=10.0, max_value=1000.0, value=80.0
        )
        garden = st.checkbox("Garden")
        terrace = st.checkbox("Terrace")
        swimming_pool = st.checkbox("Swimming pool")
        epc_score = st.selectbox(
            "EPC score",
            ["A", "B", "C", "D", "E", "F"],
            index=4,
        )
        postal_code = st.text_input("Postal code", value="1000")
        property_type = st.radio("Property type", ["apartment", "house"], index=0)
        state_of_the_property = st.selectbox(
            "State of the property",
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
            ],
            index=5,
        )
        submitted = st.form_submit_button("Predict")

    if submitted:
        # Convert postal code to latitude and longitude
        latitude, longitude = postal_code_to_latlon(postal_code)
        if np.isnan(latitude) or np.isnan(longitude):
            st.error(
                "Invalid or unknown postal code. Please enter a valid Belgian postal code."
            )
        else:
            # Build input DataFrame
            input_dict = {
                "number_of_bedrooms": [number_of_bedrooms],
                "livable_surface": [livable_surface],
                "garden": [int(garden)],
                "terrace": [int(terrace)],
                "swimming_pool": [int(swimming_pool)],
                "epc_score": [epc_score],
                "latitude": [latitude],
                "longitude": [longitude],
                "type": [property_type],
                "state_of_the_property": [state_of_the_property],
                "specific_primary_energy_consumption": [np.nan],
            }
            input_df = pd.DataFrame(input_dict)

            # Use model's preprocessing (adapted for single row)
            def preprocess_for_prediction(df):
                # One-hot encode 'type' column
                df = pd.get_dummies(df, columns=["type"], drop_first=True, dtype=int)
                if "type_house" not in df.columns:
                    df["type_house"] = 0  # Ensure both dummies always present
                # Ordinal encode 'state_of_the_property' and 'epc_score' using shared utils
                df = ordinal_encode_state(df, col="state_of_the_property")
                df = df.drop(columns=["state_of_the_property"])
                df = ordinal_encode_epc_score(df, col="epc_score")
                df = df.drop(columns=["epc_score"])
                # Ensure all columns are numeric
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                return df

            processed = preprocess_for_prediction(input_df.copy())
            st.write("Preprocessed input for model:")
            st.dataframe(processed)
            try:
                prediction = predict_price(processed)
                st.success(f"Predicted price: {int(prediction):,} €".replace(",", "."))
            except Exception as e:
                st.error(f"Prediction failed: {e}")

with tab2:
    st.header("Data Dashboard")
    # Read data from property_details_for_analysis table
    # Adjust the connection string as needed for your Airflow DB
    import os

    AIRFLOW_CONN = os.environ.get(
        "AIRFLOW__CORE__SQL_ALCHEMY_CONN",
        "postgresql+psycopg2://airflow:airflow@postgres:5432/airflow",
    )
    try:
        engine = create_engine(AIRFLOW_CONN)
        df = pd.read_sql("SELECT * FROM property_details_for_analysis", engine)
        st.subheader("Average Price by Number of Bedrooms")
        st.bar_chart(df.groupby("number_of_bedrooms")["price"].mean())
        st.subheader("Average Price by EPC Score")
        df = compute_epc_score(df, col="specific_primary_energy_consumption")
        st.bar_chart(df.groupby("epc_score")["price"].mean())

        # Boxplot: Price Distribution by State of the Property
        st.subheader("Price Distribution by State of the Property")
        import plotly.express as px

        if "state_of_the_property" in df.columns and "price" in df.columns:
            box_fig = px.box(
                df,
                x="state_of_the_property",
                y="price",
                points="outliers",
            )
            box_fig.update_xaxes(title="State of the Property", tickangle=45)
            box_fig.update_yaxes(title="Price")
            st.plotly_chart(box_fig, use_container_width=True)
        else:
            st.info("'state_of_the_property' or 'price' column not found in data.")
    except Exception as e:
        st.error(f"Failed to load data or dashboards: {e}")
