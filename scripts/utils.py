import joblib
import pandas as pd


class Setup:
    """Loads the model and required dependencies. Prepares the model for making predictions
    """

    # column names for dataframe
    __column_names = ['Sex', 'Marital status', 'Age',
                      'Education', 'Income', 'Occupation', 'Settlement size']

    def __init__(self) -> None:
        # load model
        self.__model = joblib.load("./models/k_means_6.pkl")
        # load min max scaler
        self.__scaler = joblib.load("./models/min_max_scaler.pkl")

    def predict(self, X: list[list]) -> list[str]:
        # create dataframe
        df = pd.DataFrame(X, columns=self.__column_names)
        # normalize the dataframe
        df_scaled = self.__scaler.transform(df)

        # make predictions
        predictions = self.__model.predict(df_scaled)

        # return cluster number
        return [f'Cluster {cluster}' for cluster in predictions]
