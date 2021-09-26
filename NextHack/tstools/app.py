from datetime import datetime as dt
import logging
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from tstools.config import Configuration
from tstools.model import UnitAutoML

logging.root.setLevel(logging.INFO)


class TimeSeriesML:
    def __init__(self, config: Configuration):
        self.config = config
        self.unitml = []

    @staticmethod
    def _load_file(path):
        data = pd.read_csv(path)
        if 'calls' not in data.columns:
            data['calls'] = 0
        data = data.rename(columns={'datetime': 'ds', 'calls': 'y'})
        data['ds'] = data.ds.apply(lambda x: dt.fromisoformat(x))

        return data

    def _sampler(self, split_id, base, test_base):
        data = base.copy()
        data = data[data.id == split_id].drop('id', axis=1)
        test = test_base[test_base.id == split_id].drop('id', axis=1)

        return data, test

    def _fit_predict_unit(self, split_id: int, train: DataFrame, test: DataFrame) -> DataFrame:
        automl = UnitAutoML(split_id=split_id, config=self.config)
        yhat = automl.forecast(train=train, test=test)

        dtest = test[['ds']]
        dtest['id'] = split_id
        dtest['yhat'] = yhat

        self.unitml.append(automl)

        return dtest

    def forecast(self):
        base = self._load_file(self.config.train)
        test_base = self._load_file(self.config.test)
        result = []

        for SPLIT, CUT in zip(self.config.ids, self.config.cut_off_dates):
            try:
                setattr(self.config, 'cut_off_date', CUT)

                if not self.config.enable_fs and SPLIT in [25, 56]:
                    setattr(self.config, 'from_prophet', 'middle')

                train, test = self._sampler(split_id=SPLIT, base=base, test_base=test_base)
                prediction = self._fit_predict_unit(split_id=SPLIT, train=train, test=test)
                result.append(prediction)

            except KeyboardInterrupt:
                logging.error('ERRROR! Data spitted out, if forehand')
                break

        result = pd.concat(result).reset_index(drop=True).sort_values(by=['ds', 'id'])
        result['ds'] = result['ds'].astype(str)
        result = result.rename(columns={'ds': 'datetime', 'yhat': 'calls'})

        return result

