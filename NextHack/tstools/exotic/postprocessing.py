class Split25Prep:
    def __init__(self, data, target_column, threshold):
        self.data = data
        self.target = target_column
        self.t = threshold

    def dive(self):
        data = self.data.copy()
        for i in range(1, len(data) - 1):
            if data.at[i, self.target] <= self.t and data.at[i - 1, self.target] <= self.t and \
                    data.at[i + 1, self.target] <= self.t:
                data.at[i - 1, 'zero'] = 1
                data.at[i, 'zero'] = 1
                data.at[i + 1, 'zero'] = 1

        data['zero'] = data['zero'].fillna(0)

        return data

    def sink(self):
        self.data = self.dive()
        self.data[self.target] = self.data.apply(lambda row: row[self.target] if row['zero'] == 0 else 0.0, axis=1)
        self.data = self.data.drop('zero', axis=1)

        return self.data
