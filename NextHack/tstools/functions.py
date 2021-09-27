import numpy as np


def inverse_boxcox(series, lambda_):
    return np.exp(series) if lambda_ == 0 else np.exp(np.log(lambda_ * series + 1) / lambda_)
