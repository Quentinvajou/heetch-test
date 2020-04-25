import os
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV


class TrainModel:
    def __init__(self, preprocessing=None):
        self.preprocessing = preprocessing

    def train_xgb_model(self, best_params, dtrain, dtest):
        model_construit = 0
        # Construction du modèle
        best_model = xgb.train(
            best_params,
            dtrain,
            num_boost_round=best_params['num_boost_round'],
            evals=[(dtest, "dev")],
            early_stopping_rounds=50
        )
        print("Best RMSE: {:.2f} in {} rounds".format(best_model.best_score, best_model.best_iteration+1))
        num_boost_round = best_model.best_iteration + 1
        return best_model


    def train_xgb_model_grid(self, list_parameters, Xtrain, ytrain, nb_folds=5):
        # scoring = {'msloge': 'neg_mean_squared_log_error', 'r2_score': 'r2'}
        # scoring = 'r2'
        scoring = 'neg_root_mean_squared_error'

        xgb_model = XGBRegressor()
        xgb_grid = GridSearchCV(xgb_model,
            list_parameters,
            cv = nb_folds,
            n_jobs = 5,
            scoring=scoring,
            verbose=True)

        xgb_grid.fit(Xtrain, ytrain)
        print(xgb_grid.best_score_)
        print(xgb_grid.cv_results_)
        return xgb_grid.best_params_

    def main(self):
        pass