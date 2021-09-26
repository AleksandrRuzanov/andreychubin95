import logging
import warnings
from datetime import datetime as dt
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from tqdm import tqdm
from sklearn.manifold import TSNE
from umap import UMAP
from umap.parametric_umap import ParametricUMAP
from tstools.holidays_custom.russia import Russia
from tstools.config import Configuration

from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters

logging.root.setLevel(logging.INFO)
warnings.filterwarnings('ignore')


class FeatureExtractor:
    def __init__(self, config: Configuration, time_column: str = 'ds'):
        self.config = config
        self.time_column = time_column
        self.n_components = self.config.components
        self.extraction_settings = ComprehensiveFCParameters()
        self.cat_columns = None

    def _make_holidays_features(self, data: DataFrame) -> DataFrame:
        dummy = data.copy()
        dummy['holiday'] = dummy[self.time_column].apply(lambda x: Russia().get(str(x).split(' ')[0]))

        holidays = {"Пред. Новый год": 1,
                    "Новый год": 2,
                    "День защитника отечества": 3,
                    "День женщин": 4,
                    "Праздник Весны и Труда": 5,
                    "Майские": 6,
                    "День Победы": 7,
                    "День России": 8,
                    "День народного единства": 9}

        dummy['holiday'] = dummy.holiday.apply(lambda x: 0 if x not in holidays.keys() else holidays.get(x))

        return dummy

    def _prepare_events(self, event_path: str, myevents_path: str) -> DataFrame:
        events = pd.read_csv(event_path, sep=',').rename(columns={'datetime': 'ds', 'calls': 'y'})
        events = events[(events.priority > 0) & (events.event > 0)].drop(['id', 'priority'], axis=1)
        events = events.drop_duplicates()
        events['ds'] = events.ds.apply(lambda x: dt.fromisoformat(x))
        myevents = pd.read_csv(myevents_path, sep=',')
        myevents['ds'] = myevents.ds.astype(str)
        myevents['ds'] = myevents.ds.apply(lambda x: dt.fromisoformat(x))
        data = pd.concat([events, myevents])
        data['date'] = data.ds.apply(lambda x: str(x).split(' ')[0])
        twitter = pd.read_csv(self.config.events_power, sep=',')\
            .rename(columns={'event': 'activity'})
        twitter['date'] = twitter.date.apply(lambda x: x.split(' ')[0])
        data = data.merge(twitter, on='date', how='left')
        data['activity'] = data['activity'].fillna(0.0)
        data = data.drop('date', axis=1)

        return data

    def add_events(self, dataframe: DataFrame) -> DataFrame:
        events = self._prepare_events(event_path=self.config.events, myevents_path=self.config.myevents)
        data = dataframe.copy()
        data = data.merge(events, on='ds', how='left')
        data['event'] = data['event'].fillna(0)
        data['activity'] = data['activity'].fillna(0.0)

        return data

    def add_full(self, dataframe: DataFrame) -> DataFrame:
        full = pd.read_csv(self.config.full, sep=',')
        full['ds'] = full.ds.astype(str)
        full['ds'] = full.ds.apply(lambda x: dt.fromisoformat(x))

        data = dataframe.copy()
        data['ds'] = data.ds.astype(str)
        data['ds'] = data.ds.apply(lambda x: dt.fromisoformat(x))
        data = data.merge(full, on='ds', how='left')
        data['y_all'] = data['y_all'].fillna(0.0)

        return data

    def add_cat(self, dataframe: DataFrame) -> DataFrame:
        dummy = dataframe.copy()
        dummy[self.time_column] = dummy[self.time_column].astype(str)
        dummy[self.time_column] = dummy[self.time_column].apply(lambda x: dt.strptime(x, '%Y-%m-%d %H:%M:%S'))
        dummy['hour'] = dummy[self.time_column].dt.hour

        for hour in range(24):
            dummy['hour_{}'.format(hour)] = dummy.hour.apply(lambda x: 1 if x == hour else 0)

        dummy = self._make_holidays_features(data=dummy)

        dummy['weekday'] = dummy[self.time_column].dt.weekday

        for holiday in range(1, 10):
            dummy['holiday_{}'.format(holiday)] = dummy.holiday.apply(lambda x: 1 if x == holiday else 0)

        dummy['holiday'] = dummy.holiday.apply(lambda x: 1 if x > 0 else 0)

        weekdays = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday', 7: 'Sunday'}

        for day in range(1, 8):
            dummy['{}'.format(weekdays.get(day))] = dummy.weekday.apply(lambda x: 1 if x == day else 0)

        dummy['month'] = dummy[self.time_column].dt.month

        for month in range(1, 13):
            dummy['month_{}'.format(month)] = dummy.month.apply(lambda x: 1 if x == month else 0)

        dummy['new_year'] = dummy.apply(
            lambda x: 1 if x['month'] == 1 and x[self.time_column].day in list(range(1, 10)) else 0, axis=1
        )

        dummy['covid_year'] = dummy.ds.apply(lambda x: 1 if x >= dt.fromisoformat('2020-03-01 00:00:00') else 0)

        return dummy.sort_values(by=self.time_column).drop(['month', 'weekday', 'hour'], axis=1).reset_index(drop=True)

    def add_cat_enc(self, dataframe: DataFrame) -> DataFrame:
        dummy = dataframe.copy()
        dummy[self.time_column] = dummy[self.time_column].astype(str)
        dummy[self.time_column] = dummy[self.time_column].apply(lambda x: dt.strptime(x, '%Y-%m-%d %H:%M:%S'))
        dummy['hour'] = dummy[self.time_column].dt.hour
        dummy = self._make_holidays_features(data=dummy)

        if dummy['holiday'].isna().any():
            raise AssertionError('enc_holiday contains nan-values')

        dummy['weekday'] = dummy[self.time_column].dt.weekday
        dummy['month'] = dummy[self.time_column].dt.month

        dummy['new_year'] = dummy.apply(
            lambda x: 1 if x['month'] == 1 and x[self.time_column].day in list(range(1, 10)) else 0, axis=1
        )

        dummy['covid_year'] = dummy.ds.apply(lambda x: 1 if x >= dt.fromisoformat('2020-03-01 00:00:00') else 0)

        self.cat_columns = ['hour', 'holiday', 'weekday', 'month', 'new_year', 'covid_year']

        if self.config.add_events:
            self.cat_columns.append('event')
        if self.config.impute:
            self.cat_columns.append('imputed')

        for column in self.cat_columns:
            dummy[column] = dummy[column].astype(int)

        return dummy.sort_values(by=self.time_column).reset_index(drop=True)

    @staticmethod
    def _takens_embedding(data: np.ndarray, delay: int, dimension: int) -> np.ndarray:
        if delay * dimension > len(data):
            raise NameError('Delay times dimension exceed length of data!')
        embedded_data = np.array([data[0:len(data) - delay * dimension]])
        for i in range(1, dimension):
            embedded_data = np.append(embedded_data, [data[i * delay:len(data) - delay * (dimension - i)]], axis=0)
        return embedded_data

    @staticmethod
    def pct_change(series: pd.Series, lag: int = 1) -> pd.Series:
        result = [0.0] * lag
        for i in range(lag, len(series)):
            pct = (series.iloc[i - lag] - series.iloc[i]) / series.iloc[i]
            result.append(pct)

        return pd.Series(result)

    def _reduce_dimentions_tsne(self, dataframe):
        tsne = TSNE(n_components=self.n_components, method='exact', random_state=self.config.seed, n_jobs=-1)
        logging.debug(f'Reducing dimensionality to {tsne.n_components} features')
        data = dataframe.copy()
        reduced = tsne.fit_transform(X=data.values)
        df_reduced = pd.DataFrame(data=reduced, columns=[f'tsne_{x}' for x in range(tsne.n_components)],
                                  index=data.index)

        return df_reduced

    def _reduce_dimentions_umap(self, dataframe):
        umap = UMAP(n_components=self.n_components, n_neighbors=5, random_state=self.config.seed,
                    transform_seed=self.config.seed,
                    target_metric='l2', verbose=False)
        logging.debug(f'Reducing dimensionality to {umap.n_components} features')
        data = dataframe.copy()
        reduced = umap.fit_transform(X=data.values)
        df_reduced = pd.DataFrame(data=reduced, columns=[f'tsne_{x}' for x in range(umap.n_components)],
                                  index=data.index)

        return df_reduced

    def _reduce_dimentions_param_umap(self, dataframe):
        param_umap = ParametricUMAP(n_components=self.n_components, parametric_embedding=True,
                                    random_state=self.config.seed, transform_seed=self.config.seed,
                                    target_metric='l2', n_training_epochs=10)
        logging.debug(f'Reducing dimensionality to {param_umap.n_components} features')
        data = dataframe.copy()
        reduced = param_umap.fit_transform(X=data.values)
        df_reduced = pd.DataFrame(data=reduced, columns=[f'tsne_{x}' for x in range(param_umap.n_components)],
                                  index=data.index)

        return df_reduced

    def _auto_extractor(self, dataframe: DataFrame, initial: list) -> DataFrame:
        data = dataframe.copy()
        X = extract_features(data[initial].rename(columns={x: x.replace('15T_', '15T')
                                                           for x in data.columns if '15T' in x}),
                             column_sort=self.time_column, column_id='date',
                             default_fc_parameters=self.extraction_settings,
                             impute_function=impute, n_jobs=0)

        for column in tqdm(X.columns):
            if X[column].sum() == 0.0 or len(X[column].unique()) == 1:
                X = X.drop(column, axis=1)

        logging.debug(f'Automated Feature Extractor produced {len(X.columns)} additional columns')
        
        X = self._reduce_dimentions_umap(X)
        X = X.reset_index().rename(columns={'index': 'date'})

        return X

    def _dater(self, dataframe: DataFrame) -> DataFrame:
        data = dataframe.copy()
        data['date'] = data[self.time_column].dt.date
        data['date'] = data['date'].astype(str)
        return data
    
    def _embed(self, train: DataFrame, val: DataFrame, test: DataFrame, dim: int = 4) -> (DataFrame, DataFrame, DataFrame):
        te_train = train.copy()
        te_val = val.copy()
        te_test = test.copy()

        ys = np.concatenate((train.yhat_15T_.values, val.yhat_15T_.values, test.yhat_15T_.values))
        te = self._takens_embedding(data=ys, delay=650, dimension=dim)
        
        te_train = te_train[len(ys) - len(te[0]):].copy()

        for i in range(dim):
            te_train[f'te_{i}'] = te[i][:len(te_train)]
            te_val[f'te_{i}'] = te[i][len(te_train):len(te_train)+len(te_val)]
            te_test[f'te_{i}'] = te[i][len(te_train)+len(te_val):]

        return te_train, te_val, te_test
    
    def _add_features(self, train: DataFrame, val: DataFrame, test: DataFrame) -> (DataFrame, DataFrame, DataFrame):
        ftrain = train.copy()
        fval = val.copy()
        ftest = test.copy()
        
        if self.config.add_events:
            ftrain = self.add_events(ftrain)
            fval = self.add_events(fval)
            ftest = self.add_events(ftest)
        if self.config.add_full:
            ftrain = self.add_full(ftrain)
            fval = self.add_full(fval)
            ftest = self.add_full(ftest)

        ftrain = self.add_cat_enc(ftrain)
        fval = self.add_cat_enc(fval)
        ftest = self.add_cat_enc(ftest)

        ftrain = self._dater(ftrain)
        fval = self._dater(fval)
        ftest = self._dater(ftest)

        return ftrain, fval, ftest

    def transform(self, train: DataFrame, val: DataFrame, test: DataFrame)\
            -> (DataFrame, DataFrame, DataFrame):
        fe_train = train.copy()
        fe_val = val.copy()
        fe_test = test.copy()

        fe_train, fe_val, fe_test = self._embed(fe_train, fe_val, fe_test)
        fe_train, fe_val, fe_test = self._add_features(fe_train, fe_val, fe_test)

        base = pd.concat([fe_train, fe_val, fe_test])
        base['pct_change_lag1'] = self.pct_change(series=base.yhat_15T_)
        features = self._auto_extractor(base, self.config.auto_columns)

        fe_train = fe_train.merge(features, on='date', how='left').drop('date', axis=1)
        fe_val = fe_val.merge(features, on='date', how='left').drop('date', axis=1)
        fe_test = fe_test.merge(features, on='date', how='left').drop('date', axis=1)

        return fe_train, fe_val, fe_test







