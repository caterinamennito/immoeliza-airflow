from airflow import DAG
from datetime import datetime

with DAG(
    dag_id="immoeliza_etl",
    start_date=datetime(year=2025, month=9, day=8, hour=16, minute=0),
    schedule="@daily",
    catchup=True,
    max_active_runs=1,
    render_template_as_native_obj=True
) as dag:
    from scraper.main import fetch_data
    t1 = fetch_data()
