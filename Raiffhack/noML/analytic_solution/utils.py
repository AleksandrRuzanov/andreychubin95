import numpy as np
from pandas.core.series import Series
from settings import VECTOR_COLUMNS


class Object:
    def __init__(self, series: Series, test_item: bool = False):
        self.data = series
        self.coord = (series.lat, series.lng)
        self.floor = series.floor
        self.square = series.total_square
        self.realty_type = series.realty_type
        self.date = series.date

        if not test_item:
            self.price = series.per_square_meter_price
            self.dist = series.dist
        else:
            self.price = None
            self.dist = None

        self.vector = self._build_vector

    @property
    def _build_vector(self) -> np.ndarray:
        cols = VECTOR_COLUMNS
        vector = []

        for unit in cols:
            vector.append(self.data[unit])

        return np.array([vector])
