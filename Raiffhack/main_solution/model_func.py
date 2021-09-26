import logging
import pandas as pd
from umap import UMAP, ParametricUMAP


def _reduce_dimentions_umap(dataframe):
    umap = UMAP(n_components=20, n_neighbors=5, random_state=42,
                transform_seed=42,
                target_metric='l2', verbose=False)

    logging.debug(f'Reducing dimensionality to {umap.n_components} features')
    data = dataframe.copy()
    reduced = umap.fit_transform(X=data.values)
    df_reduced = pd.DataFrame(data=reduced, columns=[f'umap_{x}' for x in range(umap.n_components)],
                              index=data.index)

    return df_reduced


def _reduce_dimentions_param_umap(dataframe):
    param_umap = ParametricUMAP(n_components=20, parametric_embedding=True,
                                random_state=42, transform_seed=42,
                                target_metric='l2', n_training_epochs=1, n_epochs=10, n_jobs=-1)
    logging.debug(f'Reducing dimensionality to {param_umap.n_components} features')
    data = dataframe.copy()
    reduced = param_umap.fit_transform(X=data.values)
    df_reduced = pd.DataFrame(data=reduced, columns=[f'pumap_{x}' for x in range(param_umap.n_components)],
                              index=data.index)

    return df_reduced