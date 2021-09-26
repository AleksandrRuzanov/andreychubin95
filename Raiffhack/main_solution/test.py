import multiprocessing
import pickle
import numpy as np
import pandas as pd
from pandarallel import pandarallel
from utils import Object
from functions import find_n_neib_by_distance, similarity_pricing
from metric import metric_loss

pandarallel.initialize(nb_workers=multiprocessing.cpu_count())


if __name__ == '__main__':
    train = pd.read_csv('/Users/andreychubin/Downloads/clean_data_v2.csv')

    with open('deviation.pickle', 'rb') as f:
        deviation = pickle.load(f)

    natural = train[train.price_type == 0].drop('price_type', axis=1).copy()
    hmade = train[train.price_type == 1].drop('price_type', axis=1).copy()


    def unit_pipeline(obj, data=train):
        obj = Object(obj, True)
        frame = data[(data.region == obj.data.region) &
                     (data.realty_type == obj.realty_type)].copy()

        neibs = find_n_neib_by_distance(frame, obj)

        return similarity_pricing(neibs, obj)

    prediction = unit_pipeline(hmade.iloc[100])

    print('Case 1')
    print(f'Prediction: {prediction}')
    print(f'Adjusted Prediction: {prediction * (1 + deviation)}')
    print(f'True: {hmade.iloc[100].per_square_meter_price}')
    print(f'Metric: {metric_loss(np.array([hmade.iloc[100].per_square_meter_price]), np.array([prediction]))}')
    print(f'Adjusted metric: {metric_loss(np.array([hmade.iloc[100].per_square_meter_price]), np.array([prediction * (1 + deviation)]))}')
    print(' ')
    prediction = unit_pipeline(natural.iloc[3])

    print('Case 2')
    print(f'Prediction: {prediction}')
    print(f'Adjusted Prediction: {prediction * (1 + deviation)}')
    print(f'True: {natural.iloc[3].per_square_meter_price}')
    print(f'Metric: {metric_loss(np.array([natural.iloc[3].per_square_meter_price]), np.array([prediction]))}')
    print(f'Adjusted metric: {metric_loss(np.array([natural.iloc[3].per_square_meter_price]), np.array([prediction * (1 + deviation)]))}')

"""
    hmade['pred_price'] = hmade.parallel_apply(unit_pipeline, axis=1)
    metric_loss(hmade['per_square_meter_price'].values, hmade['pred_price'].values)

    y_manual = hmade.per_square_meter_price.copy()
    predictions = hmade.pred_price.copy()

    deviation = ((y_manual - predictions) / predictions).median()
    corrected_price = predictions * (1 + deviation)

    with open('deviation.pickle', 'wb') as f:
        pickle.dump(deviation, f)"""
