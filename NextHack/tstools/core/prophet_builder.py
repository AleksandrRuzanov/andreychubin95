import pandas as pd
import prophet


class ProphetBuilder:
    def __init__(self):
        pass

    @staticmethod
    def get_prophet(freq: str, split_id: int) -> prophet.Prophet:
        if freq == '15T':
            holidays = pd.read_csv('/Users/andreychubin/Desktop/DS/Хакатон/holidays/holidays_15m.csv', sep=',')

            if split_id == 1:
                # 0.34266
                # 0.18
                fb = prophet.Prophet(uncertainty_samples=1000, yearly_seasonality=False,
                                     weekly_seasonality=False, daily_seasonality=False,
                                     holidays=holidays,
                                     seasonality_prior_scale=20,
                                     holidays_prior_scale=10, changepoint_prior_scale=0.005) \
                    .add_seasonality(name='monthly', period=30.5, fourier_order=60) \
                    .add_seasonality(name='weekly', period=7, fourier_order=30) \
                    .add_seasonality(name='daily', period=1, fourier_order=100) \
                    .add_seasonality(name='hourly', period=1/24, fourier_order=60) \
                    .add_seasonality(name='quarterly', period=365.25/4, fourier_order=30)

            elif split_id == 2:
                # 0.1808
                # 0.43351 or 0.3442 after cleaning
                fb = prophet.Prophet(uncertainty_samples=1000, yearly_seasonality=False,
                                     weekly_seasonality=False, daily_seasonality=False,
                                     holidays=holidays,
                                     seasonality_prior_scale=100,
                                     holidays_prior_scale=45, changepoint_prior_scale=0.001) \
                    .add_seasonality(name='monthly', period=30.5, fourier_order=15) \
                    .add_seasonality(name='weekly', period=7, fourier_order=40) \
                    .add_seasonality(name='daily', period=1, fourier_order=40) \
                    .add_seasonality(name='quarterly', period=365.25 / 4, fourier_order=10)

            elif split_id == 8:
                # 0.26413 tiny test 0.18
                fb = prophet.Prophet(uncertainty_samples=1000, yearly_seasonality=False,
                                     weekly_seasonality=False, daily_seasonality=False,
                                     holidays=holidays,
                                     seasonality_prior_scale=20,
                                     holidays_prior_scale=10, changepoint_prior_scale=0.005) \
                    .add_seasonality(name='monthly', period=30.5, fourier_order=60) \
                    .add_seasonality(name='weekly', period=7, fourier_order=30) \
                    .add_seasonality(name='daily', period=1, fourier_order=100) \
                    .add_seasonality(name='hourly', period=1 / 24, fourier_order=60) \
                    .add_seasonality(name='quarterly', period=365.25 / 4, fourier_order=30)

            elif split_id == 25:
                # 0.30836
                # 0.27 without other levels impovement
                fb = prophet.Prophet(uncertainty_samples=1000, yearly_seasonality=False,
                                     weekly_seasonality=False, daily_seasonality=False,
                                     holidays=holidays,
                                     seasonality_prior_scale=20,
                                     holidays_prior_scale=10, changepoint_prior_scale=0.005) \
                    .add_seasonality(name='monthly', period=30.5, fourier_order=15) \
                    .add_seasonality(name='weekly', period=7, fourier_order=10) \
                    .add_seasonality(name='daily', period=1, fourier_order=30) \
                    .add_seasonality(name='hourly', period=1/24, fourier_order=100) \
                    .add_seasonality(name='quarterly', period=365.25/4, fourier_order=10)

            elif split_id == 56:
                # 0.38034
                # 0.28
                fb = prophet.Prophet(uncertainty_samples=1000, yearly_seasonality=False,
                                     weekly_seasonality=False, daily_seasonality=False,
                                     holidays=holidays,
                                     seasonality_prior_scale=30,
                                     holidays_prior_scale=20, changepoint_prior_scale=0.05) \
                    .add_seasonality(name='hourly', period=1/24, fourier_order=100) \
                    .add_seasonality(name='weekly', period=7, fourier_order=20) \
                    .add_seasonality(name='daily', period=1, fourier_order=60) \
                    .add_seasonality(name='15min', period=1/96, fourier_order=30)

        elif freq == 'H':
            holidays = pd.read_csv('/Users/andreychubin/Desktop/DS/Хакатон/holidays/holidays_h.csv', sep=',')

            if split_id == 1:
                pass

            elif split_id == 2:
                pass

            elif split_id == 8:
                pass

            elif split_id == 25:
                # 0.2424
                fb = prophet.Prophet(uncertainty_samples=1000, yearly_seasonality=False,
                                     weekly_seasonality=False, daily_seasonality=False,
                                     holidays=holidays,
                                     seasonality_prior_scale=40,
                                     holidays_prior_scale=60, changepoint_prior_scale=0.005) \
                    .add_seasonality(name='monthly', period=30.5, fourier_order=15) \
                    .add_seasonality(name='weekly', period=7, fourier_order=60) \
                    .add_seasonality(name='daily', period=1, fourier_order=20) \
                    .add_seasonality(name='hourly', period=1 / 24, fourier_order=15)

            elif split_id == 56:
                pass

        return fb

    @staticmethod
    def get_default_params(freq: str) -> dict:
        seasonal_settings = {'15T': {'daily_seasonality': True,
                                     'weekly_seasonality': True,
                                     'yearly_seasonality': False},
                             'H': {'daily_seasonality': True,
                                   'weekly_seasonality': True,
                                   'yearly_seasonality': False},
                             'D': {'daily_seasonality': False,
                                   'weekly_seasonality': True,
                                   'yearly_seasonality': True},
                             'W': {'weekly_seasonality':False,
                                   'daily_seasonality': False,
                                   'yearly_seasonality': False}}

        return seasonal_settings[freq]

    @staticmethod
    def get_additional(split_id: int) -> list:
        if split_id == 1:
            return ['daily', 'daily_lower', 'daily_upper',
                    'hourly', 'hourly_lower', 'hourly_upper',
                    'monthly', 'monthly_lower', 'monthly_upper',
                    'quarterly', 'quarterly_lower', 'quarterly_upper',
                    'weekly', 'weekly_lower', 'weekly_upper']
        elif split_id == 2:
            return ['daily', 'daily_lower', 'daily_upper',
                    'monthly', 'monthly_lower', 'monthly_upper',
                    'quarterly', 'quarterly_lower', 'quarterly_upper',
                    'weekly', 'weekly_lower', 'weekly_upper']
        elif split_id == 8:
            return ['daily', 'daily_lower', 'daily_upper',
                    'hourly', 'hourly_lower', 'hourly_upper',
                    'monthly', 'monthly_lower', 'monthly_upper',
                    'quarterly', 'quarterly_lower', 'quarterly_upper',
                    'weekly', 'weekly_lower', 'weekly_upper']
        elif split_id == 25:
            return ['daily', 'daily_lower', 'daily_upper',
                    'hourly', 'hourly_lower', 'hourly_upper',
                    'monthly', 'monthly_lower', 'monthly_upper',
                    'quarterly', 'quarterly_lower', 'quarterly_upper',
                    'weekly', 'weekly_lower', 'weekly_upper']
        elif split_id == 56:
            return ['daily', 'daily_lower', 'daily_upper',
                    'hourly', 'hourly_lower', 'hourly_upper',
                    'weekly', 'weekly_lower', 'weekly_upper',
                    '15min', '15min_lower', '15min_upper']

