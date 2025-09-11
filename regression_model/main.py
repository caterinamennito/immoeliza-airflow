from utils.db_utils import get_engine
import pandas as pd

from regression_model.preprocessing import prepare_data_for_regression
from regression_model.model import trainModel
import pickle


def train_model():
    df = read_data_from_db()
    obs = list(df.columns)
    obs.remove("price")
    model = trainModel(df, obs)
    with open("regression_model/model.pkl", "wb") as f:
        pickle.dump(model, f)

def read_data_from_db():
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute("SELECT * FROM property_details_for_regression")
        return pd.DataFrame(result.fetchall(), columns=result.keys())


