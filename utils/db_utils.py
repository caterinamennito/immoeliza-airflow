from sqlalchemy import create_engine
import os
def get_engine(db_url=None):
    """
    Returns a SQLAlchemy engine for the given db_url, or the default Airflow/Postgres URL if not provided.
    """
    if db_url is None:
        #  use "postgres" when running in Docker Compose, "localhost" when running locally
        is_docker = lambda: os.path.exists('/.dockerenv')
        if is_docker():
            db_url = "postgresql+psycopg2://airflow:airflow@postgres:5432/airflow"
        else:
            db_url = "postgresql+psycopg2://airflow:airflow@localhost:5432/airflow"
    return create_engine(db_url)