import logging
from time import time
import warnings
import pandas as pd
import prophet
import gc
import pickle
from tstools.core.prophet_builder import ProphetBuilder
from tstools.config import Configuration
from pandas.core.frame import DataFrame

warnings.filterwarnings('ignore')
logging.root.setLevel(logging.INFO)


class ProphetCouncil:
    def __init__(self, split_id: int, config: Configuration,
                 prophet_methods: dict = None, from_prophet: str = 'middle'):
        self.split_id = split_id
        self.config = config
        self.from_prophet = from_prophet
        self.prophet_methods = prophet_methods
    
    @staticmethod
    def _holiday_loader(freq: str) -> DataFrame:
        if freq == '15T':
            holidays = pd.read_csv('/Users/andreychubin/Desktop/DS/Хакатон/holidays/holidays_15m.csv', sep=',')
        elif freq == 'H':
            holidays = pd.read_csv('/Users/andreychubin/Desktop/DS/Хакатон/holidays/holidays_h.csv', sep=',')
        else:
            holidays = pd.read_csv('/Users/andreychubin/Desktop/DS/Хакатон/holidays/holidays_d.csv', sep=',')
            
        return holidays
    
    def _get_fb(self, freq: str, uncertainty: int) -> prophet.Prophet:
        holidays = self._holiday_loader(freq=freq)
        fb_builder = ProphetBuilder()
        if freq == '15T' and not self.config.default_prophet:
            fb = fb_builder.get_prophet(freq='15T', split_id=self.split_id)
        else:
            params = fb_builder.get_default_params(freq=freq)
            fb = prophet.Prophet(uncertainty_samples=uncertainty, holidays=holidays, **params)
            
        return fb
    
    def _choose_columns(self, freq) -> list:
        if self.from_prophet == 'large':
            if freq != '15T':
                cols_to_take = ['ds', 'yhat', 'trend', 'yhat_lower', 'yhat_upper', 'trend_lower', 'trend_upper',
                                'additive_terms', 'additive_terms_lower', 'additive_terms_upper',
                                'daily', 'daily_lower', 'daily_upper',
                                'holidays', 'holidays_lower', 'holidays_upper',
                                'weekly', 'weekly_lower', 'weekly_upper',
                                'yearly', 'yearly_lower', 'yearly_upper',
                                'multiplicative_terms', 'multiplicative_terms_lower', 'multiplicative_terms_upper']

            else:
                cols_to_take = ['ds', 'yhat', 'trend', 'yhat_lower', 'yhat_upper', 'trend_lower', 'trend_upper',
                                'additive_terms', 'additive_terms_lower', 'additive_terms_upper',
                                'holidays', 'holidays_lower', 'holidays_upper',
                                'multiplicative_terms', 'multiplicative_terms_lower', 'multiplicative_terms_upper']

        elif self.from_prophet == 'middle':
            if freq != '15T':
                cols_to_take = ['ds', 'yhat', 'trend', 'yhat_lower', 'yhat_upper', 'trend_lower', 'trend_upper',
                                'additive_terms', 'additive_terms_lower', 'additive_terms_upper',
                                'daily', 'daily_lower', 'daily_upper',
                                'holidays', 'holidays_lower', 'holidays_upper',
                                'weekly', 'weekly_lower', 'weekly_upper']

            else:
                cols_to_take = ['ds', 'yhat', 'trend', 'yhat_lower', 'yhat_upper', 'trend_lower', 'trend_upper',
                                'additive_terms', 'additive_terms_lower', 'additive_terms_upper',
                                'holidays', 'holidays_lower', 'holidays_upper']

        elif self.from_prophet == 'small':
            cols_to_take = ['ds', 'yhat', 'trend', 'yhat_lower', 'yhat_upper', 'trend_lower', 'trend_upper']

        else:
            cols_to_take = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']

        return cols_to_take

    @staticmethod
    def _make_resampled(x: DataFrame, freq: str, how: str = 'mean') -> DataFrame:
        resampled = x[['ds', 'y']].copy()
        resampled['y'] = resampled.y.astype(float)
        if how == 'mean':
            resampled = resampled.set_index('ds').resample(freq).mean().reset_index(drop=False)
        if 'quantile' in how:
            resampled = resampled.set_index('ds').resample(freq) \
                .quantile(float(how.split('_')[1])).reset_index(drop=False)
        else:
            resampled = resampled.set_index('ds').resample(freq).median().reset_index(drop=False)

        resampled = resampled.dropna()
        return resampled
           
    def _forecast(self, train: DataFrame, test: DataFrame, periods: int, freq: str,
                  uncertainty: int = 1000, add: str = '') -> (DataFrame, DataFrame):
        train_fb = train.copy()
        test_fb = test.copy()

        try:
            if freq == '15T' and not self.config.default_prophet:
                path = f'/Users/andreychubin/Desktop/DS/tstools/pickles/fb_{self.split_id}_{self.config.fb_how}.pickle'
                with open(path, 'rb') as file:
                    fb = pickle.load(file)
                logging.info('Prophet was read from pickle')
            else:
                raise FileNotFoundError
        except FileNotFoundError:
            fb = self._get_fb(freq=freq, uncertainty=uncertainty)
            fb.fit(train_fb)
            if freq == '15T' and not self.config.default_prophet:
                path = f'/Users/andreychubin/Desktop/DS/tstools/pickles/fb_{self.split_id}_{self.config.fb_how}.pickle'
                with open(path, 'wb') as file:
                    pickle.dump(fb, file)
                logging.debug('Prophet was saved to pickle')

        predictions = fb.make_future_dataframe(periods=periods*2, freq=freq)
        forecast = fb.predict(predictions)

        if self.from_prophet != 'all':
            cols_to_take = self._choose_columns(freq=freq)
            if freq == '15T':
                cols_to_take = cols_to_take + ProphetBuilder().get_additional(split_id=self.split_id)
        else:
            cols_to_take = forecast.columns

        forecast = forecast[[x for x in cols_to_take if x in forecast.columns]]

        forecast = forecast.rename(columns={x: f'{x}_{freq}_{add}' for x in forecast.columns if x not in ['ds', 'y']})

        if 'y' in forecast.columns:
            forecast = forecast.drop('y', axis=1)

        forecast_train = forecast.merge(train_fb, on='ds', how='inner')
        forecast_test = forecast.merge(test_fb, on='ds', how='inner')

        gc.collect()

        return forecast_train, forecast_test
    
    @staticmethod
    def _connector(x, level: str) -> str:
        if level == 'H':
            return f'{x.year}-{x.month}-{x.day}-{x.hour}'
        elif level == 'D':
            return f'{x.year}-{x.month}-{x.day}'
        elif level == 'W':
            return f'{x.year}-{x.month}-{x.isocalendar()[1]}'
        elif level == 'M':
            return f'{x.year}-{x.month}'

    def _append_con(self, x: DataFrame, y: DataFrame, level: str) -> DataFrame:
        data = x.copy()
        dummy = y.copy()
        data['connector'] = data.ds.apply(lambda x: self._connector(x, level))
        dummy['connector'] = dummy.ds.apply(lambda x: self._connector(x, level))
        data = data.merge(dummy.drop(['ds', 'y'], axis=1), on='connector', how='left').drop('connector', axis=1)

        return data
        
    def _append_level(self, data: (DataFrame, DataFrame), train: DataFrame, test: DataFrame, level: str,
                      resample: str = 'mean', uncertainty: int = 1000) -> (DataFrame, DataFrame):
        base_train = data[0].copy()
        base_test = data[1].copy()

        train_resampled = self._make_resampled(train, level, how=resample)
        test_resampled = self._make_resampled(test, level, how=resample)

        periods = len(test_resampled)
        logging.debug(f'make {level} forecast, periods: {periods}')

        dummy_train, dummy_test = self._forecast(train=train_resampled, test=test_resampled, periods=periods,
                                                 freq=level, add=resample, uncertainty=uncertainty)

        base_train = self._append_con(base_train, dummy_train, level)
        base_test = self._append_con(base_test, dummy_test, level)

        return base_train, base_test

    def forecast(self, train: DataFrame, test: DataFrame) -> (DataFrame, DataFrame):
        logging.info(f'Implementing Prophet')

        super_start = time()
        p_15min = len(test)
        start = time()
        logging.debug(f'make 15 min forecast, periods: {p_15min}')
        base_train, base_test = self._forecast(train=train, test=test, periods=p_15min, freq='15T')
        logging.debug(f'forecasted in: {round(time() - start, 4)} sec.')

        if self.prophet_methods is not None:
            logging.debug('Appending levels')
            for method in self.prophet_methods.keys():
                logging.debug(f'using method: {method}')
                for level in self.prophet_methods[method]:
                    start = time()
                    base_train, base_test = \
                        self._append_level(data=(base_train, base_test), train=train,
                                           test=test, level=level, resample=method)
                    logging.debug(f'forecasted in: {round(time() - start, 4)} sec.')

        gc.collect()

        logging.info(f'Prophet made multilevel forecast in {round(time() - super_start, 4)} sec.')

        return base_train, base_test
