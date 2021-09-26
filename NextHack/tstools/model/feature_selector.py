import pandas as pd
from pandas.core.frame import DataFrame
from tstools.model import OptunaTuner
from sklearn.preprocessing import MinMaxScaler


def feature_selection_by_feature_importance(opt: OptunaTuner, train: DataFrame, val: DataFrame) -> list:
    opt.fit(train=train, val=val)
    fi = pd.DataFrame({'feature': [x for x in train.columns if x not in ['ds', 'y']],
                       'importance': opt.judge.feature_importance()})
    features = list(fi[fi.importance > 0].feature.values)
    return features


def feature_selection_by_corr(train: DataFrame) -> list:
    data = train.drop('ds', axis=1).copy()
    scaler = MinMaxScaler()
    X = scaler.fit_transform(data)
    data = pd.DataFrame(data=X, columns=data.columns, index=data.index)
    corrmat = data.corr()
    corrmat = round(corrmat, 3)
    top_corr_features = corrmat.index
    corr = data[top_corr_features].corr()
    importance = corr[['y']].sort_values('y', ascending=False)[(corr.y >= 0.09) | (corr.y <= -0.09)].copy()
    features = importance.index.tolist()[1:]

    return features
