import logging
from time import time
from datetime import datetime as dt
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from tstools.core import ProphetCouncil
from tstools.model import OptunaTuner, OptunaTunerCV
from tstools.model.classifier import ClassifierLeaf
from tstools.preprocessing import TimeSeriesProcessing, FeatureExtractor, EventsCleaner
from tstools.config import Configuration
from tstools.model.feature_selector import feature_selection_by_corr
from tstools.functions import inverse_boxcox

logging.root.setLevel(logging.INFO)


class UnitAutoML:
    def __init__(self, split_id: int, config: Configuration):
        self.config = config
        self.split_id = split_id
        self.preprocessing = TimeSeriesProcessing(config=self.config)
        self.prophet = ProphetCouncil(split_id=split_id,
                                      config=self.config,
                                      prophet_methods=self.config.prophet_methods,
                                      from_prophet=self.config.from_prophet)
        self.optuna = OptunaTuner(split_id=split_id, config=self.config)
        self.classifier = ClassifierLeaf()
        self.feature_extractor = FeatureExtractor(config=self.config)
        self._test = None
        self._start = time()
        self._zeroes = None

    @staticmethod
    def _clean(x: np.ndarray) -> np.ndarray:
        result = []
        for value in x:
            if value >= 0:
                result.append(value)
            else:
                result.append(0.0)
        return np.array(result)

    def _prepare(self, dataframe: DataFrame) -> DataFrame:
        data = dataframe.copy()
        if self.config.no_events or self.config.no_holidays:
            logging.debug(f'Cleaning events {self.config.no_events}, holidays {self.config.no_holidays}')
            ec = EventsCleaner(split_id=self.split_id, config=self.config)
            if self.config.no_events:
                data = ec.no_events(dataframe=data)
            if self.config.no_holidays:
                data = ec.no_holidays(dataframe=data)

        data = self.preprocessing.transform(dataframe=data)
        return data

    def _prophet(self, train: DataFrame, test: DataFrame) -> (DataFrame, DataFrame, DataFrame):
        f_train, p_test = self.prophet.forecast(train=train, test=test)

        p_train = f_train.iloc[:-self.config.validation_size]
        p_val = f_train.iloc[-self.config.validation_size:]

        return p_train, p_val, p_test

    def _classifier(self, train: DataFrame, val: DataFrame, test: DataFrame) -> np.ndarray:
        data = pd.concat([train, val]).reset_index(drop=True)
        if self.config.detrend:
            data['y'] = np.rint(self.preprocessing.detrender.inverse_transform(data['y']))
        top = pd.DataFrame({'value': data.y.value_counts(normalize=True).index,
                            'frac': data.y.value_counts(normalize=True).values})\
            .sort_values(by='frac', ascending=False)

        if top.frac.iloc[0] < 0.3:
            c_val = np.array([1.0]*len(val))
            c_test = np.array([1.0]*len(test))
            self.classifier.set(c_val, c_test)
            return np.array([1.0]*len(test))

        else:
            setattr(self.classifier, "value", top.value.iloc[0])
            setattr(self.classifier, "cat_columns", self.feature_extractor.cat_columns)
            inv_train = data.iloc[:len(train)].copy()
            inv_val = data.iloc[len(train):].copy()
            y_pred = self.classifier.fit_predict(inv_train, inv_val, test)
            return y_pred

    @staticmethod
    def _apply_classification_results(array: np.ndarray, zeroes: np.ndarray) -> np.ndarray:
        result = []
        for x, zero in zip(list(array), list(zeroes)):
            if zero == 1.0:
                result.append(x)
            else:
                if x > 2:
                    result.append(x)
                else:
                    result.append(zero)
        return np.array(result)

    @staticmethod
    def _inverse_boxcox(series, lambda_):
        return np.exp(series) if lambda_ == 0 else np.exp(np.log(lambda_ * series + 1) / lambda_)

    def _resample(self, dataframe: DataFrame) -> DataFrame:
        data = dataframe.copy()
        if self.config.detrend:
            data['y'] = np.rint(self.preprocessing.detrender.inverse_transform(data['y']))
        top = pd.DataFrame({'value': data.y.value_counts(normalize=True).index,
                            'frac': data.y.value_counts(normalize=True).values})\
            .sort_values(by='frac', ascending=False)

        if top.frac.iloc[0] >= 0.3:
            data['bin'] = self.classifier.binarizer(data['y'])  # now value is set to 0
            dummy = data[((data.ds <= dt.fromisoformat('2020-02-01 00:00:00')) & (data.y == 0))]\
                .sample(frac=0.5, random_state=self.config.seed).copy()
            data = data[~((data.ds <= dt.fromisoformat('2020-02-01 00:00:00')) & (data.y == 0))]
            """
            window = 96*30
            count = int(len(data)//window)
            new_data = []
            for i in range(0, count):
                if i != count:
                    dummy = data.iloc[i*window:(i+1)*window]
                else:
                    dummy = data.iloc[i*window:]
                dummy_non_zero = dummy[dummy.bin != 0].copy()
                dummy_zero = dummy[dummy.bin == 0].sample(frac=0.8, random_state=self.config.seed).copy()
                dummy = pd.concat([dummy_non_zero, dummy_zero])
                new_data.append(dummy.drop('bin', axis=1))

            resampled = pd.concat(new_data).sort_values(by='ds').reset_index(drop=True)
            """
            resampled = pd.concat([dummy, data])\
                .sort_values(by='ds')\
                .reset_index(drop=True)\
                .drop('bin', axis=1)

            logging.info(f'Data was downsampled from {len(dataframe)} to {len(resampled)} rows')

            if self.config.detrend:
                resampled['y'] = self.preprocessing.detrender.transform(resampled['y'])

            return resampled

        else:
            return dataframe

    def _apply(self, train: DataFrame, test: DataFrame) -> None:
        logging.info(f'---------------------- SPLIT {self.split_id} ---------------------')

        # Preprocessing
        dtrain = self._prepare(dataframe=train)

        # Prophet
        dtrain, dval, dtest = self._prophet(dtrain, test)

        # Feature Selection
        if self.config.enable_fs:
            features = ['ds'] + feature_selection_by_corr(dtrain) + ['y']
        else:
            features = dtrain.columns

        # Feature Extraction
        if 'imputed' in features:
            dtest['imputed'] = 0

        dtrain, dval, dtest = self.feature_extractor.transform(dtrain[features], dval[features], dtest[features])

        # Remove 'imputed' if exists
        """
        if 'imputed' in dtrain.columns:
            dtrain = dtrain[dtrain.imputed == 0].drop('imputed', axis=1)
        if 'imputed' in dval.columns:
            dval = dval[dval.imputed == 0].drop('imputed', axis=1)
        if 'imputed' in dtest.columns:
            dtest = dtest.drop('imputed', axis=1)
        """
        # Resample if needed
        if self.config.resample:
            dtrain_resampled = self._resample(dtrain)
        else:
            dtrain_resampled = dtrain.copy()

        # Optuna + LGBM Regression
        self.optuna.fit(train=dtrain_resampled, val=dval, cat_columns=self.feature_extractor.cat_columns)

        # Test Dataset
        self._test = dtest[self.optuna.features]

        # Optuna + LGBM Classifier
        if self.config.classifier:
            self._zeroes = self._classifier(dtrain, dval, self._test)

    def _predict(self) -> np.ndarray:
        x_test = self._test.copy()
        # Prediction
        y_hat = self.optuna.predict(x_test)
        # Reverse boxcox
        if self.config.boxcox:
            y_hat = inverse_boxcox(pd.Series(y_hat), self.config.lmbda).values
        # Reverse detrending
        if self.config.detrend:
            y_hat = self.preprocessing.detrender.inverse_transform(pd.Series(y_hat)).values
        # Clean Negative
        y_hat = self._clean(y_hat)
        # Apply Classification Results
        if self.config.classifier:
            y_hat = self._apply_classification_results(y_hat, self._zeroes)

        logging.info(f'Prediction made in {round((time() - self._start)/60, 4)} min')

        return y_hat

    def forecast(self, train: DataFrame, test: DataFrame) -> np.ndarray:
        self._apply(train, test)
        yhat = self._predict()
        return yhat


class UnitAutoMLCV(UnitAutoML):
    def __init__(self, split_id: int, config: Configuration):
        super().__init__(split_id, config)
        self.optuna = OptunaTunerCV(split_id=split_id, config=self.config)

    def _predict(self) -> np.ndarray:
        logging.info('Started Inference')
        x_test = self._test.copy()
        y_hats = self.optuna.predict(x_test)
        predictions = pd.DataFrame({f'yhat{i}': y_hats[i] for i in range(len(y_hats))})
        if self.optuna.solver is not None:
            yhat = self.optuna.solver.predict(predictions)
        else:
            yhat = predictions.apply(lambda row: row.mean(), axis=1).values
        yhat = self._clean(yhat)

        logging.info('Prediction made')
        logging.info('####################END####################')

        return yhat
