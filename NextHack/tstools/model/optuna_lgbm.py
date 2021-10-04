import logging
import warnings
from time import time

import numpy as np
import optuna
import optuna.integration.lightgbm as lgb
import pandas as pd
from optuna.integration.lightgbm import Dataset
from optuna.integration.sklearn import OptunaSearchCV
from pandas.core.frame import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit

from tstools.config import Configuration
from tstools.functions import inverse_boxcox

optuna.logging.set_verbosity(optuna.logging.ERROR)
logging.root.setLevel(logging.INFO)
warnings.filterwarnings('ignore')


class OptunaTuner:
    def __init__(self, split_id: int,
                 config: Configuration,
                 boosting_type: str = 'gbdt',
                 importance_type: str = 'split',):
        self.split_id = split_id
        self.config = config
        self.boosting_type = boosting_type
        self.importance_type = importance_type
        self.cat_columns = None
        self.train = None
        self.val = None
        self.judge = None
        self.features = None
        self._val = None
        self._test = None

    def _datasets(self) -> (Dataset, Dataset):
        x_train = self.train.drop(['ds', 'y'], axis=1)
        y_train = self.train.y.values
        self.features = list(x_train.columns)
        x_test = self.val[self.features].copy()
        y_test = self.val.y.values
        dtrain = Dataset(x_train, label=y_train, categorical_feature=self.cat_columns)
        dval = Dataset(x_test, label=y_test, categorical_feature=self.cat_columns)

        return dtrain, dval

    def _wape(self, preds, train_data):
        if isinstance(train_data, Dataset):
            y_true = train_data.label
        else:
            y_true = train_data

        y_pred = preds
        corr_fact = np.where(y_true == 0, 1, y_true)
        corr_forecast = np.where((y_true == 0) & (y_pred == 0), 1, y_pred)

        if self.split_id in [1, 2, 8]:
            wape = abs(corr_fact - corr_forecast).sum() / corr_fact.sum()
            if wape > 1:
                return 'wape', 1, False

            return 'wape', wape, False

        else:
            if y_pred.sum() == 0:
                if y_true.sum() == 0:
                    return 'wape', 0, False
                else:
                    return 'wape', 1, False

            acc_day = np.where((corr_fact + corr_forecast) == 0, 1, 0)
            acc_day = np.where((corr_fact >= corr_forecast) & (corr_fact != 0), corr_forecast / corr_fact, acc_day)
            acc_day = np.where((corr_fact < corr_forecast) & (corr_forecast != 0), corr_fact / corr_forecast, acc_day)
            wape = 1 - (acc_day * corr_fact).sum() / corr_fact.sum()

            return 'wape', wape, False

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

        metric = 'mae' if self.config.detrend or self.config.boxcox else 'wape'

        params = {"n_estimators": 1000,
                  "objective": "regression",
                  "verbosity": -1,
                  "metric": metric,
                  "learning_rate": 0.001,
                  "boosting_type": self.boosting_type,
                  "random_state": self.config.seed}

        logging.debug('Optuna started hyperopt search...')

        start = time()

        self.judge = lgb.tuner.train(params, dtrain, valid_sets=[dtrain, dval],
                                     verbose_eval=0, early_stopping_rounds=500,
                                     feval=self._wape)

        logging.debug(f'Hyperopts found in: {round(time()-start, 4)} sec')

    def _test_prediction(self) -> None:
        x_test = self.val[self.features].copy()
        y_test = self.val.y.values
        prediction = self.judge.predict(x_test, num_iteration=self.judge.best_iteration)

        if self.config.detrend:
            prediction = self.config.detrender.inverse_transform(pd.Series(prediction)).values

        if self.config.boxcox:
            prediction = inverse_boxcox(pd.Series(prediction), self.config.lmbda).values
            y_test = inverse_boxcox(pd.Series(y_test), self.config.lmbda).values

        prediction = self._clean(prediction)

        wape = self._wape(prediction, y_test)

        best_params = self.judge.params
        logging.debug("Best params:", best_params)
        logging.info("Validation WAPE = {}".format(wape[1]))

        self._val = prediction

    def fit(self, train: DataFrame, val: DataFrame, cat_columns: list = 'auto') -> None:
        self.cat_columns = cat_columns
        self.train = train.copy()
        self.val = val.copy()

        self._optuna_train()
        self._test_prediction()

    def predict(self, test: DataFrame) -> np.ndarray:
        yhat = self.judge.predict(test, num_iteration=self.judge.best_iteration)
        self._test = yhat
        return yhat

    def get_leaf_data(self) -> (np.ndarray, np.ndarray):
        return self._val, self._test


# work in progress
class OptunaTunerCV(OptunaTuner):
    def __init__(self, split_id: int, config: Configuration):
        super().__init__(split_id, config)
        self.solver = None

    def _dataset(self) -> Dataset:
        base = pd.concat([self.train, self.val])
        x_train = base.drop(['ds', 'y'], axis=1)
        y_train = base.y.values
        self.features = list(x_train.columns)
        dtrain = Dataset(x_train, label=y_train, categorical_feature=self.cat_columns)

        return dtrain

    def _optuna_train(self) -> None:
        dtrain, dval = self._datasets()

        params = {"n_estimators": 500,
                  "objective": "regression",
                  "verbosity": -1,
                  "metric": "wape",
                  "boosting_type": self.boosting_type,
                  "random_state": self.config.seed}

        logging.debug('Optuna started hyperopt search...')

        start = time()

        tuner = lgb.LightGBMTunerCV(params=params, train_set=dtrain, verbose_eval=0, early_stopping_rounds=200,
                                    feval=self._wape, folds=TimeSeriesSplit(n_splits=5),
                                    return_cvbooster=True)
        tuner.run()
        self.judge = tuner.get_best_booster()

        logging.debug(f'Hyperopts found in: {round(time() - start, 4)} sec')

    def _rf(self, X, y):
        reg = RandomForestRegressor(n_estimators=200, criterion='mae', random_state=self.config.seed)
        param_distributions = {"min_samples_leaf": optuna.distributions.IntUniformDistribution(2, 200),
                               "min_samples_split": optuna.distributions.IntUniformDistribution(2, 5),
                               "max_depth": optuna.distributions.IntUniformDistribution(5, 200),
                               "ccp_alpha": optuna.distributions.LogUniformDistribution(0.001, 100)}
        optuna_search = OptunaSearchCV(reg, param_distributions, n_jobs=-1, n_trials=200, verbose=1, cv=2)
        optuna_search.fit(X, y)
        self.solver = optuna_search
        y_pred = optuna_search.predict(X)
        return y_pred

    def _lin(self, X, y):
        reg = LinearRegression(n_jobs=-1)
        reg.fit(X, y)
        self.solver = reg
        y_pred = reg.predict(X)
        return y_pred

    def _test_prediction(self) -> None:
        x_test = self.val[self.features].copy()
        y_test = self.val.y.values
        y = y_test
        prediction = self.judge.predict(x_test, num_iteration=self.judge.best_iteration)
        X = pd.DataFrame({f'yhat{i}': prediction[i] for i in range(len(prediction))})

        if self.config.cv_solver == 'rf':
            pred = self._rf(X, y)
        else:
            pred = self._lin(X, y)

        pred = self._clean(pred)
        best_params = self.judge.params
        logging.debug("Best params:", best_params)
        logging.info("  WAPE = {}".format(self._wape(pred, y_test)[1]))
