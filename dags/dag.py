from airflow import DAG
from airflow.decorators import task
from airflow.operators.bash import BashOperator
from datetime import datetime


with DAG(
    dag_id="immoeliza_etl",
    start_date=datetime(year=2025, month=9, day=8, hour=16, minute=0),
    schedule="@hourly",
    catchup=False,
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
    def prepare_data_for_regression():
        from regression_model.preprocessing import prepare_data_for_regression

        prepare_data_for_regression()

    @task
    def train_model_task():
        from regression_model.main import train_model

        train_model()

    dvc_version_model = BashOperator(
        task_id="dvc_version_model",
        bash_command="""
            dvc add regression_model/model.pkl
            git add regression_model/model.pkl.dvc
            git commit -m 'Version new trained model with DVC'
            dvc push
        """,
    )

    t1 = fetch_data_apartment()
    t2 = fetch_data_house()
    t3 = prepare_data_for_analysis_task()
    t4 = prepare_data_for_regression()
    t5 = train_model_task()

    t1 >> [t3, t4]
    t2 >> [t3, t4]
    t4 >> t5
    t5 >> dvc_version_model
