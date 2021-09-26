import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.preprocessing import MinMaxScaler


class Scaler:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()

    def fit_transform_data(self, dataframe: DataFrame) -> DataFrame:
        data = dataframe.copy()
        X = self.scaler.fit_transform(data)
        result = pd.DataFrame(data=X, columns=data.columns, index=data.index)

        return result

    def fit_transform_target(self, array: np.ndarray) -> np.ndarray:
        shape = array.shape
        scaled = self.target_scaler.fit_transform(array.reshape(-1, 1))
        return scaled.reshape(shape[0],)

    def transform_data(self, dataframe: DataFrame) -> DataFrame:
        data = dataframe.copy()
        X = self.scaler.transform(data)
        result = pd.DataFrame(data=X, columns=data.columns, index=data.index)
        return result

    def transform_target(self, array: np.ndarray) -> np.ndarray:
        shape = array.shape
        scaled = self.target_scaler.transform(array.reshape(-1, 1))
        return scaled.reshape(shape[0],)

    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        if isinstance(y, list):
            y = np.array(y)
        shape = y.shape
        scaled = self.target_scaler.inverse_transform(y.reshape(-1, 1))
        return scaled.reshape(shape[0],)
