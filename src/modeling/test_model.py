import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from scipy.stats import ttest_ind, normaltest, ks_2samp
from sklearn.metrics import explained_variance_score, r2_score
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.model_selection import cross_validate


class ModelTest:
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
        metrics['variance_explained'] = explained_variance_score(test_y, yhat)
        metrics['r2'] = r2_score(test_y, yhat)

        metrics_dist = self.compare_dist(train_y, test_y)
        metrics = {**metrics, **metrics_dist}

        test_residual = test_y - yhat
        _, pvalue, _, _ = het_breuschpagan(test_residual.loc[~test_x.isna().any(axis=1)],
                                           test_x[~test_x.isna().any(axis=1)])  ## null hypothesis (homoscedasticity)
        metrics['homoscedasticity_pv'] = pvalue

        shifting_features = self.covariate_shift_test(train_x, test_x)
        metrics['shifting_features_dict'] = shifting_features
        return metrics