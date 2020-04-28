import os
import xgboost as xgb
from xgboost import XGBClassifier
import pickle
import pandas as pd
from datetime import datetime
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GridSearchCV, train_test_split

from src.infrastructure.settings import DATASET_SAMPLING_FRACTION


class TrainModel:
    def __init__(self, dict_modeling_params, preprocessing=None, test_model=None):
        self.preprocessing = preprocessing
        self.default_parameters = dict_modeling_params['default_parameters']
        self.list_of_parameters = dict_modeling_params['list_of_parameters']
        self.look_for_best_params = dict_modeling_params['look_for_best_params']
        self.test_model = test_model

    def run_training(self):
        df, is_preprocessed = self.preprocessing.load_and_merge_datasets(frac=DATASET_SAMPLING_FRACTION)
        if not is_preprocessed:
            df = self.preprocessing.prepare_dataset_for_training(df, save_processed_dataset=True)
        df = self.preprocessing.filter_dataset_for_training_column_wise(df)

        train_x, test_x, train_y, test_y = train_test_split(df.iloc[:, :-1], df.iloc[:, -1],
                                                            test_size=.3, random_state=42)

        ''' This was tested to check if outliers removal would increase accuracy. it doesn't
        train_cleaned = self.preprocessing.filter_dataset_for_training_row_wise(pd.concat([train_x, train_y], axis=1))
        train_x = train_cleaned.iloc[:, :-1]
        train_y = train_cleaned.iloc[:, -1]
        '''
        sampler = RandomOverSampler(sampling_strategy='minority')
        # sampler = RandomUnderSampler(sampling_strategy='majority')
        train_x, train_y = sampler.fit_resample(train_x, train_y)

        dtrain = xgb.DMatrix(train_x, label=train_y)
        dtest = xgb.DMatrix(test_x, label=test_y)
        best_params = self.default_parameters
        if self.look_for_best_params:
            best_params = self.train_xgb_model_grid(self.list_of_parameters, train_x, train_y, nb_folds=2)
        best_params['num_boost_round'] = self.default_parameters['num_boost_round']

        best_model = self.train_xgb_model(best_params, dtrain, dtest)
        yhat = best_model.predict(xgb.DMatrix(test_x))

        metrics = self.test_model.test_model_and_dataset(train_x, train_y, test_x, test_y, yhat)
        metrics['data_size'] = df.shape[0]
        return best_model, metrics

    def train_xgb_model(self, best_params, dtrain, dtest):
        # Construction du modèle
        best_model = xgb.train(
            best_params,
            dtrain,
            num_boost_round=best_params['num_boost_round'],
            evals=[(dtest, "dev")],
            early_stopping_rounds=50
        )
        print("Best logloss: {:.2f} in {} rounds".format(best_model.best_score, best_model.best_iteration+1))
        num_boost_round = best_model.best_iteration + 1
        return best_model

    def train_xgb_model_grid(self, list_parameters, Xtrain, ytrain, nb_folds=2):
        scoring = 'f1'

        xgb_model = XGBClassifier()
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

    def save_model_and_metrics(self, model, metrics):
        path_output_model = 'data/models/'
        models_saved = [f for f in os.listdir(path_output_model) if os.path.isfile(os.path.join(path_output_model, f))]
        os.system("mkdir -p " + path_output_model)
        version = len(models_saved)
        pickle.dump(model, open(path_output_model + 'model_' + str(version+1) + '.pkl', "wb"))

        self.save_metrics(model, metrics, version)

    def save_metrics(self, model, metrics, version):
        metrics_file = 'data/trusted/metrics.csv'

        df_features = pd.DataFrame(model.get_fscore().items(), columns=['feature', 'fscore'])
        df_features['pourcentage'] = round((df_features.fscore / df_features.fscore.sum()) * 100, 2)
        df_features = df_features[['feature', 'pourcentage']]
        dict_feature = df_features.set_index('feature')['pourcentage'].to_dict()

        metrics['logloss'] = model.best_score
        metrics['features'] = dict_feature
        metrics['model'] = 'model_' + str(version+1)
        metrics['timestamp'] = datetime.now()
        if os.path.isfile(metrics_file) and os.access(metrics_file, os.R_OK):
            df_metrics = pd.read_csv(metrics_file)
            df_metrics = df_metrics.append([metrics])
            df_metrics.to_csv(metrics_file, index=False)
        else:
            df_metrics = pd.DataFrame([metrics])
            df_metrics.to_csv(metrics_file, index=False)

    def main(self):
        pass