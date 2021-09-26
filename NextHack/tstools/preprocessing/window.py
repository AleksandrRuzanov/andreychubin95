import pandas as pd
from tqdm import trange


class WindowMaker:
    def __init__(self, window, time_column='ds', target='yhat_15T_'):
        self.window = window
        self.time_column = time_column
        self.target = target
        self.data = None

    def _make(self):
        df = self.data[[self.time_column, self.target]].reset_index(drop=True).T.copy()
        dates = []
        values = []
        for i in trange(len(self.data) - self.window):
            line = df.loc[self.time_column, self.window + i]
            dates.append(line)
            values.append(df.loc[self.target, i:self.window + i].values)

        return dates, values

    def embeddings(self):
        pass

    def make(self, dataframe: pd.DataFrame, merge=True):
        self.data = dataframe.copy()
        dates, values = self._make()
        data = pd.DataFrame(data=values)
        target_col = data.columns[-1]
        data = data.drop(target_col, axis=1)
        data = data.rename(columns={x: f'lag_{x}' for x in data.columns if isinstance(x, int)})
        data[self.time_column] = dates

        if merge:
            data = self.data.merge(data, on=[self.time_column], how='inner')

        return data
