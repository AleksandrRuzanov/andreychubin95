
def search(self):
    result = pd.DataFrame(columns=['split', 'with_events', 'with_holidays', 'scaling', 'param', 'wape'])
    param0 = None
    for param in self.params:
        if param != param0:
            print('New Run started')
            wape = self.start(param=param)
            print('Run ended')
            for key, value in param.items():
                print("    {}: {}".format(key, value))

            print(f'got WAPE: {wape}')

            result = result.append({'split': self.split,
                                    'with_events': self.with_events,
                                    'with_holidays': self.with_holidays,
                                    'scaling': self.scaling,
                                    'param': str(param),
                                    'wape': wape}, ignore_index=True)

        param0 = param

    try:
        dummy = pd.read_csv(f'try_{self.split}.csv', sep=';')
        result = pd.concat([dummy, result])
    except FileNotFoundError:
        pass

    result.to_csv(f'try_{self.split}.csv', header=True, index=False, sep=';')