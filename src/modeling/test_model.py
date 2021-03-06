import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from scipy.stats import ttest_ind, normaltest, ks_2samp
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import cross_validate


class TestModel:
    def __init__(self):
        pass

    def compare_distribution(self, dist_a, dist_b):
        metrics_dist = {}
        _, pvalue_a = normaltest(dist_a)
        _, pvalue_b = normaltest(dist_b)
        metrics_dist['normal_dist_pv'] = np.mean((pvalue_a, pvalue_b))
        if pvalue_a <= 0.05 and pvalue_b <= 0.05:  # check if both dist are Normal
            _, pvalue_ind = ttest_ind(dist_a, dist_b,
                                      equal_var=False)  # Welch's test -> here we reject null hypo (averages
        elif pvalue_a > 0.05 and pvalue_b > 0.05:
            _, pvalue_ind = ks_2samp(dist_a, dist_b)
        else:
            pvalue_ind = None
        metrics_dist['equal_dist_pv'] = pvalue_ind
        return metrics_dist

    def covariate_shift_test(self, train_x, test_x):
        train_set = train_x[~train_x.isna().any(axis=1)]
        test_set = test_x[~test_x.isna().any(axis=1)]
        train_set.loc[:, 'origin'] = 'train'
        test_set.loc[:, 'origin'] = 'test'
        train_set = train_set.sample(test_set.shape[0], random_state=43)
        test_set = test_set.sample(test_set.shape[0], random_state=42)

        ds = pd.concat([train_set, test_set], axis=0)
        ds = ds.sample(frac=1)
        target = ds['origin']
        ds.drop('origin', axis=1, inplace=True)

        model = RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_leaf=5)
        shifting_features = {}
        for col in ds.columns:
            score = cross_validate(model, pd.DataFrame(ds[col]), target, cv=2, scoring='roc_auc')
            mean_test_score = np.mean(score['test_score'])
            if mean_test_score > .8:
                shifting_features[col] = mean_test_score
        return shifting_features

    def test_model_and_dataset(self, train_x, train_y, test_x, test_y, yhat):
        metrics = {}
        predictions = [round(value) for value in yhat]

        metrics['f1_score'] = f1_score(test_y, predictions)
        metrics['precision_score'] = precision_score(test_y, predictions)
        metrics['recall_score'] = recall_score(test_y, predictions)
        metrics['accuracy_score'] = accuracy_score(test_y, predictions)
        metrics['balanced_accuracy_score'] = balanced_accuracy_score(test_y, predictions)
        metrics['ratio_T_in_train'] = train_y.value_counts(normalize=True)[True]
        metrics['ratio_T_in_test'] = test_y.value_counts(normalize=True)[True]
        return metrics
