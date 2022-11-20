import numpy as np
import pandas as pd
import os
from shapely.geometry import shape
from functools import reduce
import json

from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
from sklearn.metrics import recall_score, make_scorer
from sklearn.utils import shuffle
import lightgbm as lgbm

def preprocess(data, train=True):
    
    def parse_json(obj):
        coords = shape(json.loads(obj.values[0])).bounds
        x = (coords[0] + coords[2]) / 2
        y = (coords[1] + coords[3]) / 2
        return (x, y)

    target_col = pd.Index(['crop'])
    area_col = pd.Index(['area'])
    geo_col = pd.Index(['.geo'])
    ts_cols = data.columns.difference(area_col.append([geo_col, target_col])).to_list()
    
    # Mode
    if train:
        target = data[target_col]
        features = data.drop(target_col.to_list(), axis=1)
    else:
        target = None
        features = data
        
    features_ts = features[ts_cols].copy()
    
    features_geo = features[geo_col].copy()
    coordinates = features_geo.apply(parse_json, axis=1)
    features_geo =  pd.DataFrame(coordinates.to_list(), columns=['x', 'y'], index=features_geo.index)
    
    features_area = features[area_col].copy()
    features_parts = [features_ts, features_geo, features_area]

    features = reduce(lambda left, right: left.join(right), features_parts)
    return features, target


def main():
    train_file = 'data/train_dataset_train.csv'
    
    data = pd.read_csv(train_file)
    data.sort_index(axis=1, inplace=True)
    data.set_index('id', inplace=True)
    features, target = preprocess(data, rename=True)
    features_cols = features.columns

    data = features.join(target)
    data = shuffle(data)
    features = data[features_cols]
    target = data.crop
    
    features_train, features_val, target_train, target_val = train_test_split(features, target, train_size = 0.8, stratify = target, random_state = 2022)
    split_index = [-1 if idx in features_train.index else 0 for idx in features.index]
    pds = PredefinedSplit(test_fold = split_index)
    
    scoring_f = lambda y_true, y_pred: recall_score(y_true, y_pred, average="macro", zero_division=0)
    scoring = make_scorer(scoring_f, greater_is_better=True)
    boosting_clf = lgbm.LGBMClassifier(n_estimators=1000, n_jobs=-1)
    
    parameters = {
        'boosting_type':    ['goss', 'gbdt', 'dart', 'rf'],
        'learning_rate':    [0.05, 0.01, 0.1],
        'num_leaves':       [15, 31, 42],
        'max_depth' :       [5, 10, 25],
        'min_data_in_leaf': [15, 20, 25],
        'feature_fraction': [0.7, 0.8, 0.9],
        'reg_alpha':        [0, 0.5],
        'reg_lambda':       [0, 0.5],
    }


    grid = GridSearchCV(boosting_clf,
                        parameters, 
                        cv=pds,
                        n_jobs=-1,
                        scoring=scoring,
                        verbose=1)

    result = grid.fit(features, target)
    
    print('best score:', result.best_score_)
    print('best params:', result.best_params_)
    
if __name__ == "__main__":
    main()