from sqlalchemy import create_engine

def get_engine(db_url=None):
    """
    Returns a SQLAlchemy engine for the given db_url, or the default Airflow/Postgres URL if not provided.
    """
    if db_url is None:
        db_url = "postgresql+psycopg2://airflow:airflow@localhost:5432/airflow"
    return create_engine(db_url)