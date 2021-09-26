import numpy as np


def hit_func(deviation: float) -> float:
    w = 1.1
    if deviation < -0.6:
        return 9 * w
    elif -0.6 <= deviation < -0.15:
        return w * ((1 + (deviation / 0.15)) ** 2)
    elif -0.15 <= deviation < 0.15:
        return 0
    elif 0.15 <= deviation < 0.6:
        return ((deviation / 0.15) - 1) ** 2
    else:
        return 9


def metric_loss(y_true, y_pred):
    assert y_true.shape == y_pred.shape, "Shapes do not match"

    result = []

    for true, pred in zip(y_true, y_pred):
        deviation = (true - pred) / pred
        res = hit_func(deviation)
        result.append(res)

    return np.array(result).mean()


if __name__ == '__main__':
    a = np.array([1, 2, 3])
    b = np.array([2, 3, 4])

    metric(a, b)
