import numpy as np
import pandas as pd
import os
from shapely.geometry import shape
from functools import reduce
import json

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import recall_score, precision_score
from sklearn.utils import shuffle
from sklearn.multiclass import OutputCodeClassifier
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
	test_file = 'data/test_dataset_test.csv'

	data = pd.read_csv(train_file)
	data.sort_index(axis=1, inplace=True)
	data.set_index('id', inplace=True)
	features, target = preprocess(data, rename=True)
	features_cols = features.columns

	data = features.join(target)
	data = shuffle(data)
	features = data[features_cols]
	target = data.crop

	best_params = {'boosting': 'goss', 
				   'feature_fraction': 0.9, 
				   'learning_rate': 0.01, 
				   'max_depth': 10, 
				   'min_data_in_leaf': 15, 
				   'num_leaves': 31, 
				   'reg_alpha': 0, 
				   'reg_lambda': 0}

	# Train + Validation

	skf = StratifiedKFold(n_splits = 4)
	skf.get_n_splits(features.values, target.values)
	clf = OutputCodeClassifier(lgbm.LGBMClassifier(n_estimators=1000, **best_params), code_size=4)
	scoring = lambda estimator, x, y: recall_score(y, estimator.predict(x), average="macro", zero_division=0)
	score = cross_val_score(clf, features.values, target.values, cv=skf, scoring=scoring)

	print('Mean cross val score': score.mean())

	# Prediction

	test_data = pd.read_csv(test_file)
	test_data.sort_index(axis=1, inplace=True)
	test_data.set_index('id', inplace=True)

	test_features, _ = preprocess(test_data, train=False)

	clf = OutputCodeClassifier(lgbm.LGBMClassifier(n_estimators=1000, **best_params), code_size=4)
	clf.fit(features.values, target.values)

	test_target = clf.predict(test_features.values)
	result = pd.DataFrame(columns=['id', 'crop'])
	result['id'] = test_features.index
	result['crop'] = test_target
	result.to_csv('results.csv', index=False)

if __name__ == '__main__':
	main()

