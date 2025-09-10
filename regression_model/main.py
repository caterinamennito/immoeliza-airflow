from regression_model.preprocessing import prepare_data_for_regression
from regression_model.model import trainModel
import pickle

if __name__ == "__main__":
    df = prepare_data_for_regression()
    model = trainModel(df)
    with open("regression_model/model.pkl", "wb") as f:
        pickle.dump(model, f)
