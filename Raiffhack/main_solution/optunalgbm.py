import logging
import warnings
from time import time

import numpy as np
import optuna
import optuna.integration.lightgbm as lgb
from optuna.integration.lightgbm import Dataset
import pandas as pd
from pandas.core.frame import DataFrame

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer

optuna.logging.set_verbosity(optuna.logging.ERROR)
logging.root.setLevel(logging.INFO)
warnings.filterwarnings('ignore')


NUMERIC = ['osm_amenity_points_in_0.001',
       'osm_amenity_points_in_0.005', 'osm_amenity_points_in_0.0075',
       'osm_amenity_points_in_0.01', 'osm_building_points_in_0.001',
       'osm_building_points_in_0.005', 'osm_building_points_in_0.0075',
       'osm_building_points_in_0.01', 'osm_catering_points_in_0.001',
       'osm_catering_points_in_0.005', 'osm_catering_points_in_0.0075',
       'osm_catering_points_in_0.01', 'osm_city_closest_dist', 'osm_city_nearest_population',
       'osm_crossing_closest_dist', 'osm_crossing_points_in_0.001',
       'osm_crossing_points_in_0.005', 'osm_crossing_points_in_0.0075',
       'osm_crossing_points_in_0.01', 'osm_culture_points_in_0.001',
       'osm_culture_points_in_0.005', 'osm_culture_points_in_0.0075',
       'osm_culture_points_in_0.01', 'osm_finance_points_in_0.001',
       'osm_finance_points_in_0.005', 'osm_finance_points_in_0.0075',
       'osm_finance_points_in_0.01', 'osm_healthcare_points_in_0.005',
       'osm_healthcare_points_in_0.0075', 'osm_healthcare_points_in_0.01',
       'osm_historic_points_in_0.005', 'osm_historic_points_in_0.0075',
       'osm_historic_points_in_0.01', 'osm_hotels_points_in_0.005',
       'osm_hotels_points_in_0.0075', 'osm_hotels_points_in_0.01',
       'osm_leisure_points_in_0.005', 'osm_leisure_points_in_0.0075',
       'osm_leisure_points_in_0.01', 'osm_offices_points_in_0.001',
       'osm_offices_points_in_0.005', 'osm_offices_points_in_0.0075',
       'osm_offices_points_in_0.01', 'osm_shops_points_in_0.001',
       'osm_shops_points_in_0.005', 'osm_shops_points_in_0.0075',
       'osm_shops_points_in_0.01', 'osm_subway_closest_dist',
       'osm_train_stop_closest_dist', 'osm_train_stop_points_in_0.005',
       'osm_train_stop_points_in_0.0075', 'osm_train_stop_points_in_0.01',
       'osm_transport_stop_closest_dist', 'osm_transport_stop_points_in_0.005',
       'osm_transport_stop_points_in_0.0075',
       'osm_transport_stop_points_in_0.01',
       'reform_count_of_houses_1000', 'reform_count_of_houses_500',
       'reform_house_population_1000', 'reform_house_population_500',
       'reform_mean_floor_count_1000', 'reform_mean_floor_count_500',
       'reform_mean_year_building_1000', 'reform_mean_year_building_500', 'total_square']

ENC = ['osm_city_nearest_name', 'realty_type', 'region']


class OptunaTuner:
    def __init__(self, n_estimators: int, metric: str = 'raiff_metric',
                 cat_columns: str = 'auto', seed: int = 42, val_size: float = 0.2):
        self.n_estimators = n_estimators
        self.metric = metric
        self.cat_columns = cat_columns
        self.seed = seed
        self.val_size = val_size
        self.scaler = StandardScaler()
        self.label = []
        self.prerpocessor = ColumnTransformer(
            transformers=[('num', self.scaler, NUMERIC)])
        self.train = None
        self.val = None
        self.model = None
        self.features = None

    def _datasets(self) -> (Dataset, Dataset):
        x_train = self.train.drop('target', axis=1)
        y_train = self.train.target.values
        self.features = list(x_train.columns)
        x_test = self.val[self.features].copy()
        y_test = self.val.target.values
        dtrain = Dataset(x_train, label=y_train, categorical_feature=self.cat_columns)
        dval = Dataset(x_test, label=y_test, categorical_feature=self.cat_columns)

        return dtrain, dval

    @staticmethod
    def _hit_func(deviation: float) -> float:
        w = 1.1
        if deviation < -0.6:
            return 9 * w
        elif -0.6 <= deviation < -0.15:
            return w * ((1 + (deviation / 0.15)) ** 2)
        elif -0.15 <= deviation < 0.15:
            return 0
        elif 0.15 <= deviation < 0.6:
            return ((deviation / 0.15) - 1) ** 2
        else:
            return 9

    def _metric(self, y_true, y_pred):
        assert y_true.shape == y_pred.shape, "Shapes do not match"

        result = []

        for true, pred in zip(y_true, y_pred):
            deviation = (true - pred) / pred
            res = self._hit_func(deviation)
            result.append(res)

        return np.array(result).mean()

    def _raiff_metric(self, preds, train_data):
        if isinstance(train_data, Dataset):
            y_true = train_data.label
        else:
            y_true = train_data

        y_pred = preds

        metric = self._metric(y_true, y_pred)

        return 'raiff_metric', metric, False

    @staticmethod
    def _clean(x: np.ndarray) -> np.ndarray:
        result = []
        for value in list(x):
            if value >= 0:
                result.append(value)
            else:
                result.append(0.0)
        return np.array(result)

    def _optuna_train(self) -> None:
        dtrain, dval = self._datasets()

        params = {"n_estimators": self.n_estimators,
                  "objective": "regression",
                  "verbosity": -1,
                  "metric": self.metric,
                  "learning_rate": 0.01,
                  "boosting_type": 'gbdt',
                  "random_state": self.seed}

        logging.debug('Optuna started hyperopt search...')

        start = time()

        self.model = lgb.tuner.train(params, dtrain, valid_sets=[dtrain, dval],
                                     verbose_eval=0, early_stopping_rounds=100,
                                     feval=self._raiff_metric)

        logging.debug(f'Hyperopts found in: {round(time()-start, 4)} sec')

    def _test_prediction(self) -> None:
        x_test = self.val[self.features].copy()
        y_test = self.val.target.values
        prediction = self.model.predict(x_test, num_iteration=self.model.best_iteration)
        metric = self._raiff_metric(prediction, y_test)

        best_params = self.model.params
        logging.debug("Best params:", best_params)
        logging.info(f"Validation Raiff_metric = {metric[1]}")

    def fit(self, X: DataFrame, y: np.ndarray) -> None:
        X_prep = self.prerpocessor.fit_transform(X)
        X_prep = pd.DataFrame(data=X_prep, columns=NUMERIC)

        for col in NUMERIC:
            X[col] = X_prep[col].values

        for enc_col in ENC:
            encoder = LabelEncoder()
            X[enc_col] = encoder.fit_transform(X[enc_col])
            self.label.append(encoder)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.val_size, random_state=self.seed)

        self.train = X_train.copy()
        self.train['target'] = y_train

        self.val = X_val.copy()
        self.val['target'] = y_val

        self._optuna_train()
        self._test_prediction()

    def predict(self, test: DataFrame) -> np.ndarray:
        test_prep = self.prerpocessor.transform(test)
        test_prep = pd.DataFrame(data=test_prep, columns=NUMERIC)
        for col in NUMERIC:
            test[col] = test_prep[col].values

        for num, enc_col in enumerate(ENC):
            test[enc_col] = self.label[num].transform(test[enc_col])

        yhat = self.model.predict(test[self.features], num_iteration=self.model.best_iteration)
        return yhat
