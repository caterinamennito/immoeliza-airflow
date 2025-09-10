from airflow import DAG
from airflow.decorators import task
from datetime import datetime

with DAG(
    dag_id="immoeliza_etl",
    start_date=datetime(year=2025, month=9, day=8, hour=16, minute=0),
    schedule="@daily",
    catchup=True,
    max_active_runs=1,
    render_template_as_native_obj=True,
) as dag:

    @task
    def fetch_data_apartment():
        from scraper.main import fetch_data

        fetch_data(property_type="apartment")

    @task
    def fetch_data_house():
        from scraper.main import fetch_data

        fetch_data(property_type="house")

    @task
    def prepare_data_for_analysis_task():
        from data_analysis.main import prepare_data_for_analysis

        prepare_data_for_analysis()

    @task
    def prepare_data_for_regression_task():
        from regression_model.main import prepare_data_for_regression

        prepare_data_for_regression()

    # t1 = fetch_data_apartment()
    # t2 = fetch_data_house()
    # t3 = prepare_data_for_analysis_task()
    t4 = prepare_data_for_regression_task()

    # t1 >> [t3, t4]
    # t2 >> [t3, t4]

