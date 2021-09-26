import multiprocessing
import pickle
import pandas as pd
from pandarallel import pandarallel
from utils import Object
from settings import FILL_COLUMNS
from functions import find_n_neib_by_distance, similarity_pricing

pandarallel.initialize(nb_workers=multiprocessing.cpu_count())


if __name__ == '__main__':
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    test['per_square_meter_price'] = 0
    test = test.set_index('id', drop=True)
    test = test[train.columns]

    for elem in FILL_COLUMNS:
        med = test[elem].median()
        test[elem] = test[elem].fillna(med)

    with open('deviation.pickle', 'rb') as f:
        deviation = pickle.load(f)


    def unit_pipeline(obj, data=train):
        obj = Object(obj, True)
        frame = data[(data.region == obj.data.region) &
                     (data.realty_type == obj.realty_type)].copy()

        neibs = find_n_neib_by_distance(frame, obj)

        return similarity_pricing(neibs, obj)


    test['per_square_meter_price'] = test.parallel_apply(unit_pipeline, axis=1)
    test['per_square_meter_price'] = test['per_square_meter_price'] * (1 + deviation)
    test = test.reset_index(drop=False)
    test[['id', 'per_square_meter_price']].to_csv('submission.csv', index=False)
