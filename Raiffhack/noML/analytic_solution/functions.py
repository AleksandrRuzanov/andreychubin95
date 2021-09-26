import pandas as pd
from scipy import spatial
from pandas.core.frame import DataFrame
from geopy.distance import distance
from utils import Object


def find_n_neib_by_distance(frame: DataFrame, obj: Object) -> list:
    df = frame.copy()
    df['dist'] = frame.apply(lambda row: distance(obj.coord, (row['lat'], row['lng'])).km, axis=1)
    df = df.sort_values(by='dist').iloc[1:101]
    df['neib'] = df.apply(lambda row: Object(row), axis=1)

    return df['neib'].to_list()


def similarity_pricing(_list, obj):
    frame = pd.DataFrame(columns=['price', 'cosine'])

    for unit in _list:
        sim = spatial.distance.cosine(obj.vector, unit.vector)
        frame = frame.append({'price': unit.price, 'cosine': sim}, ignore_index=True)

    frame = frame.sort_values(by='cosine')
    frame = frame.iloc[:50].copy()
    frame['cosine'] = 1 - (frame.cosine / frame.cosine.sum())
    frame['cosine'] = frame.cosine / frame.cosine.sum()
    result_price = (frame.price * frame.cosine).sum()

    return result_price
