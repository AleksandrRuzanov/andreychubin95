import datetime


class Configuration:
    def __init__(self, ids: list,
                 cut_off_dates: list,
                 validation_size: int,
                 no_events=False,
                 no_holidays=False,
                 add_events=True,
                 add_full=False,
                 train: str = '/Users/andreychubin/Desktop/DS/tstools/data/train.csv',
                 test: str = '/Users/andreychubin/Desktop/DS/tstools/data/test.csv',
                 clean_outliers: bool = True,
                 impute: bool = True,
                 boxcox: bool = True,
                 detrend: bool = False,
                 classifier: bool = False,
                 prophet_methods: dict = None,
                 from_prophet: str = 'all',
                 set_default_prophet: bool = False,
                 holidays_15m: str = '/Users/andreychubin/Desktop/DS/tstools/data/holidays/holidays_15m.csv',
                 holidays_h: str = '/Users/andreychubin/Desktop/DS/tstools/data/holidays/holidays_h.csv',
                 holidays_d: str = '/Users/andreychubin/Desktop/DS/tstools/data/holidays/holidays_d.csv',
                 events: str = '/Users/andreychubin/Desktop/DS/tstools/data/events.csv',
                 myevents: str = '/Users/andreychubin/Desktop/DS/tstools/data/myevents.csv',
                 events_power: str = '/Users/andreychubin/Desktop/DS/tstools/data/future_events.csv',
                 full: str = '/Users/andreychubin/Desktop/DS/tstools/data/full.csv',
                 auto_columns: list = None,
                 resample: bool = True,
                 cut_off_date: datetime.datetime = None,
                 enable_feature_selection: bool = False,
                 components_of_dim_reduction: int = 100,
                 random_state: int = 42,
                 enable_cv: bool = False,
                 cv_solver: str = 'lin'):
        # Main data
        self.ids = ids
        self.cut_off_dates = cut_off_dates
        self.validation_size = validation_size
        # Clean Params
        self.no_events = no_events
        self.no_holidays = no_holidays
        self.add_events = add_events
        self.add_full = add_full
        self.clean_outliers = clean_outliers
        self.impute = impute
        self.boxcox = boxcox
        self.detrend = detrend
        self.classifier = classifier
        # Prophet params
        self.prophet_methods = prophet_methods

        if self.prophet_methods is None:
            self.prophet_methods = {
                'mean': ['H', 'D', 'W'],
                'median': ['H', 'D', 'W']
            }

        self.from_prophet = from_prophet
        self.default_prophet = set_default_prophet
        # Paths to files
        self.train = train
        self.test = test
        self.holidays_15m = holidays_15m
        self.holidays_h = holidays_h
        self.holidays_d = holidays_d
        self.events = events
        self.myevents = myevents
        self.events_power = events_power
        self.full = full
        # FE Param
        self.auto_columns = auto_columns

        if self.auto_columns is None:
            self.auto_columns = ['yhat_15T_', 'yhat_lower_15T_', 'yhat_upper_15T_'] + ['ds', 'date']

        # Rest
        self.resample = resample
        self.cut_off_date = cut_off_date
        self.enable_fs = enable_feature_selection
        self.components = components_of_dim_reduction
        self.seed = random_state
        self.enable_cv = enable_cv
        self.cv_solver = cv_solver

        if self.default_prophet:
            self.fb_how = f'default_{self.no_events}_{self.no_holidays}_{self.clean_outliers}_{self.impute}_{self.boxcox}'
        else:
            self.fb_how = f'custom_{self.no_events}_{self.no_holidays}_{self.clean_outliers}_{self.impute}_{self.boxcox}'

        # Keep
        self.detrender = None
        self.val_y = None
        self.lmbda = None


if __name__ == '__main__':
    config = Configuration(ids=list(range(5)), cut_off_dates=list(range(5)),
                           validation_size=100)

    setattr(config, 'cut_off_date', datetime.datetime.fromisoformat('2021-08-06 00:00:00'))
    setattr(config, 'abs_cut', 900)

    print(config.cut_off_date)
