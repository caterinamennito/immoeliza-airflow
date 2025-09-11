from regression_model.preprocessing import prepare_data_for_regression
from regression_model.model import trainModel
import pickle

def prepare_data_for_regression_and_train_model():
    df = prepare_data_for_regression()
    obs = list(df.columns)
    obs.remove("price")
    model = trainModel(df, obs)
    with open("regression_model/model.pkl", "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    prepare_data_for_regression_and_train_model()

