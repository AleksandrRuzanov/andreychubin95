import pandas as pd
from optunalgbm import OptunaTuner, NUMERIC, ENC
import pickle


if __name__ == '__main__':
    points = ['lat', 'lng']
    TARGET = 'per_square_meter_price'

    train = pd.read_csv('/Users/andreychubin/Downloads/train.csv')
    test = pd.read_csv('/Users/andreychubin/Downloads/test.csv')

    train = train.set_index('id', drop=True)
    test = test.set_index('id', drop=True)

    train_0 = train[train.price_type == 0].drop('price_type', axis=1).copy()
    train_1 = train[train.price_type == 1].drop('price_type', axis=1).copy()

    X_train = train_0[ENC + NUMERIC + points].copy()
    y_train = train_0[TARGET].values

    tuner = OptunaTuner(n_estimators=500, cat_columns=ENC)

    tuner.fit(X_train, y_train)

    with open('tuner.pickle', 'wb') as f:
        pickle.dump(tuner, f)
