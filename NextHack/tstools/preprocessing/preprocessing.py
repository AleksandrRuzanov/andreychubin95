import logging
import pandas as pd
from datetime import datetime as dt
from pandas.core.frame import DataFrame
from scipy.stats import boxcox
from tstools.config import Configuration
from tstools.preprocessing.detector import AnomalyDetector
from sktime.transformations.series.detrend import Detrender
from sktime.forecasting.trend import PolynomialTrendForecaster

logging.root.setLevel(logging.INFO)


class EventsCleaner:
    def __init__(self, split_id: int, config: Configuration):
        self.split = split_id
        self.config = config

    def no_events(self, dataframe: DataFrame) -> DataFrame:
        events = pd.read_csv(self.config.events, sep=',')
        events = events[(events.priority > 0) & (events.id == self.split)]\
            .drop('id', axis=1)\
            .rename(columns={'datetime': 'ds'})
        events['ds'] = events.ds.apply(lambda x: dt.strptime(x, '%Y-%m-%d %H:%M:%S'))
        events['event'] = events.event.astype(int)
        data = dataframe.copy()
        data = data.merge(events, on='ds', how='left')
        data = data.fillna(0)
        data = data[data.event == 0].drop(['event', 'priority'], axis=1)

        logging.debug(f'After cleaning events dataset reduced to {len(data)} rows')

        return data

    def no_holidays(self, dataframe: DataFrame) -> DataFrame:
        holidays = pd.read_csv(self.config.holidays_15m, sep=',')
        holidays['ds'] = holidays.ds.apply(lambda x: dt.fromisoformat(x))
        data = dataframe.copy()
        ex = data.merge(holidays[['ds', 'holiday']], on='ds', how='inner')
        data = data[~data.index.isin(ex.index)].reset_index(drop=True)

        logging.debug(f'After holidays dataset reduced to {len(data)} rows')

        return data


class TimeSeriesProcessing:
    def __init__(self, config: Configuration, time_column: str = 'ds', target_column: str = 'y'):
        self.config = config
        self.time = time_column
        self.target = target_column
        self.detrender = None

    @staticmethod
    def _anomaly_params():
        params = {'threshold': 5.0, 'drift': 2.0}
        return params

    def _remove_anomalies(self, dataframe):
        data = dataframe.copy()
        params = self._anomaly_params()
        detector = AnomalyDetector(backward_window_size=7 * 96, forward_window_size=2 * 96, **params)
        anomalies = detector.detect(data[self.target])
        data['anomalies'] = anomalies
        data = data[data['anomalies'] != 1].drop('anomalies', axis=1)

        return data

    def _add_missing_dates(self, dataframe):
        data = dataframe.copy()
        total = pd.date_range(start=data[self.time].min(), end=data[self.time].max(), freq='15T')
        dummy = pd.DataFrame(data=total, columns=[self.time])
        dummy = dummy.merge(data, on=self.time, how='left')

        return dummy

    def _impute_missing(self, dataframe):
        data = dataframe.copy()
        data['imputed'] = data[self.target].isna().astype(int)
        data[self.target] = data[self.target].interpolate(method='linear').apply(lambda x: 0.0 if x < 0.0 else x)

        return data

    def _detrend(self, series, degree: int = 2):
        ptf = PolynomialTrendForecaster(degree=degree)
        detrender = Detrender(ptf)
        detrended = detrender.fit_transform(series)
        self.detrender = detrender

        return detrended

    def _boxcox(self, frame):
        data = frame.copy()
        data[self.target] = data[self.target] + 0.00000001
        data[self.target], lambda_prophet = boxcox(data[self.target])
        setattr(self.config, 'lmbda', lambda_prophet)
        return data

    def transform(self, dataframe: DataFrame) -> DataFrame:
        data = dataframe.copy()

        if self.config.cut_off_date is not None:
            data = data[data[self.time] >= self.config.cut_off_date]
        if self.config.clean_outliers:
            data = self._remove_anomalies(data)
        if self.config.impute:
            logging.debug('Imputing missing data')
            data = self._add_missing_dates(data)
            data = self._impute_missing(data)

        data = data.reset_index(drop=True)

        setattr(self.config, "val_y", data.y.values[-self.config.validation_size:])

        if self.config.boxcox:
            data = self._boxcox(data)

        if self.config.detrend:
            logging.debug('Detrending time series')
            data[self.target] = self._detrend(data[self.target])
            setattr(self.config, "detrender", self.detrender)

        return data


if __name__ == '__main__':
    from tstools.legacy.dataset import create_default_dataset

    conf = Configuration(ids=[1], cut_off_dates=[dt.fromisoformat("2020-12-01 00:00:00")],
                         validation_size=96 * 30 * 3)

    test = create_default_dataset(size=50000)

    tsp = TimeSeriesProcessing(conf)
    output = tsp.transform(test)

    print(output.y.iloc[:20].values)

