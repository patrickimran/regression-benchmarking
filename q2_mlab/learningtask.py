import json
import numpy as np
import pandas as pd
import time

# CV Methods
from sklearn.model_selection import RepeatedStratifiedKFold

# Metrics
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score,
                roc_auc_score, precision_recall_curve, auc, f1_score)

# Algorithms
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier, Ridge
from xgboost import XGBRegressor, XGBClassifier
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import (GradientBoostingClassifier,
                              GradientBoostingRegressor)
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.mixture import BayesianGaussianMixture

from abc import ABC


class LearningTask(ABC):
    algorithms = {}

    def __init__(
        self, table, metadata, algorithm, params, n_repeats, distance_matrix
    ):
        self.distance_matrix = distance_matrix
        self.params = json.loads(params)
        self.X = table.transpose().matrix_data
        self.metadata = metadata
        self.y = self.metadata.to_numpy()
        self.learner = self.algorithms[algorithm]
        self.cv_idx = 0
        self.idx = 0
        self.n_repeats = n_repeats

        # Preallocate lists in size n_repeats * len(y) * size(float)
        self.table_size = self.n_repeats * self.y.shape[0]
        self.results = {}
        self.results["CV_IDX"] = np.empty(self.table_size, dtype=int)
        self.results["SAMPLE_ID"] = np.empty(self.table_size, dtype=object)
        self.results["Y_PRED"] = np.empty(self.table_size, dtype=float)
        self.results["Y_TRUE"] = np.empty(self.table_size, dtype=object)
        self.results["RUNTIME"] = np.empty(self.table_size, dtype=float)

        # TODO Validate the shapes of X and y, sample_id agreement
        # TODO Validate y is of type int
        # If checks fail, throw exception and end - handled in preprocess

    def contains_nan(self, y_pred):
        if (np.any(pd.isnull(y_pred))) or (not np.all(np.isfinite(y_pred))):
            return True
        else:
            return False

    def tabularize(self):
        results_table = pd.DataFrame.from_dict(self.results)
        return results_table

    def print_attributes(self):  # TODO delete
        print(self.learner)
        print(self.params)


class ClassificationTask(LearningTask):

    algorithms = {
        "KNeighborsClassifier": KNeighborsClassifier,
        "RandomForestClassifier": RandomForestClassifier,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "XGBClassifier": XGBClassifier,
        "RidgeClassifier": RidgeClassifier,
        "LinearSVC": LinearSVC,
    }

    def __init__(
        self, table, metadata, algorithm, params, n_repeats, distance_matrix
    ):
        super().__init__(
            table, metadata, algorithm, params, n_repeats, distance_matrix
        )

        kfold = RepeatedStratifiedKFold(5, self.n_repeats, random_state=2020)
        self.splits = kfold.split(self.X, self.y)

        # Lists because they must be one-dimensional to be converted to
        # pd.DataFrame columns
        self.results["Y_PROB"] = [] * self.table_size
        self.results["ACCURACY"] = np.empty(self.table_size, dtype=float)
        self.results["AUPRC"] = np.empty(self.table_size, dtype=float)
        self.results["AUROC"] = np.empty(self.table_size, dtype=float)
        self.results["F1"] = np.empty(self.table_size, dtype=float)

    def cv_fold(self, train_index, test_index):
        X_train, X_test = self.X[train_index], self.X[test_index]
        y_train, y_test = self.y[train_index], self.y[test_index]
        y_test_ids = self.metadata.index[test_index]

        use_probabilities = False
        if hasattr(self.learner, 'predict_proba'):
            use_probabilities = True

        # Start timing
        start = time.process_time()
        m = self.learner(**self.params)
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        # End timimg
        end = time.process_time()

        runtime = end - start
        nrows = len(y_pred)
        curr_indices = list(range(self.idx, self.idx + nrows))

        if use_probabilities:
            probas = m.predict_proba(X_test)
        else:
            probas = [np.nan] * nrows

        self.results["RUNTIME"][curr_indices] = [runtime] * nrows
        self.results["CV_IDX"][curr_indices] = [self.cv_idx] * nrows
        self.results["Y_PRED"][curr_indices] = y_pred
        self.results["Y_TRUE"][curr_indices] = y_test_ids
        self.results["SAMPLE_ID"][curr_indices] = y_test_ids
        self.results["Y_PROB"].extend(probas)
        '''
        Catch-22:
        Can only index a numpy array with a list (as seen above). 1D numpy
        arrays cannot contain tuples (probas is a tuple of length n_classes)
        so the array must be init as a multidim array (nrows, n_classes).
        Multidim arrays cannot be converted into columns of a dataframe.

        current fix:
        Make any tuple columns as lists, and dynamically allocate them.
        '''

        if self.contains_nan(y_pred):
            # All null
            self.results["AUPRC"][curr_indices] = [np.nan] * nrows
            self.results["AUROC"][curr_indices] = [np.nan] * nrows
            self.results["ACCURACY"][curr_indices] = [np.nan] * nrows
            self.results["F1"][curr_indices] = [np.nan] * nrows
        elif not use_probabilities:
            # Just F1 and Accuracy
            acc_score = accuracy_score(y_pred, y_test)
            self.results["ACCURACY"][curr_indices] = [acc_score] * nrows
            f1 = f1_score(y_test, y_pred)
            self.results["F1"][curr_indices] = [f1] * nrows

            # Others null
            self.results["AUPRC"][curr_indices] = [np.nan] * nrows
            self.results["AUROC"][curr_indices] = [np.nan] * nrows
        elif use_probabilities:
            # All metrics.
            mean_recall = np.linspace(0, 1, 50)
            probas = probas[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, probas)
            self.results["AUPRC"][curr_indices] = [
                auc(recall, precision)
                ] * nrows
            self.results["AUROC"][curr_indices] = [
                roc_auc_score(y_test, probas)
            ] * nrows
            acc_score = accuracy_score(y_pred, y_test)
            self.results["ACCURACY"][curr_indices] = [acc_score] * nrows
            f1 = f1_score(y_test, y_pred)
            self.results["F1"][curr_indices] = [f1] * nrows

        self.cv_idx += 1
        self.idx += nrows


class RegressionTask(LearningTask):

    algorithms = {
        "KNeighborsRegressor": KNeighborsRegressor,
        "RandomForestRegressor": RandomForestRegressor,
        "GradientBoostingRegressor": GradientBoostingRegressor,
        "XGBRegressor": XGBRegressor,
        "LinearSVR": LinearSVR,
        "RidgeRegressor": Ridge,
    }

    def __init__(
        self, table, metadata, algorithm, params, n_repeats, distance_matrix
    ):
        super().__init__(
            table, metadata, algorithm, params, n_repeats, distance_matrix
        )

        # TODO replace kfold with repeated SORTED stratified
        kfold = RepeatedStratifiedKFold(5, self.n_repeats, random_state=2020)
        self.splits = kfold.split(self.X, self.y)

        self.results["RMSE"] = np.empty(self.table_size, dtype=float)
        self.results["R2"] = np.empty(self.table_size, dtype=float)

    def cv_fold(self, train_index, test_index):
        X_train, X_test = self.X[train_index], self.X[test_index]
        y_train, y_test = self.y[train_index], self.y[test_index]
        y_test_ids = self.metadata.index[test_index]

        # Start timing
        start = time.process_time()
        m = self.learner(**self.params)
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        # End timimg
        end = time.process_time()

        runtime = end - start
        nrows = len(y_pred)
        curr_indices = list(range(self.idx, self.idx + nrows))

        self.results["RUNTIME"][curr_indices] = [runtime] * nrows
        self.results["CV_IDX"][curr_indices] = [self.cv_idx] * nrows
        self.results["Y_PRED"][curr_indices] = y_pred
        self.results["Y_TRUE"][curr_indices] = y_test_ids
        self.results["SAMPLE_ID"][curr_indices] = y_test_ids

        if not self.contains_nan(y_pred):
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            self.results["RMSE"][curr_indices] = [rmse] * nrows
            r2 = r2_score(y_test, y_pred)
            self.results["R2"][curr_indices] = [r2] * nrows
        else:
            self.results["RMSE"][curr_indices] = [None] * nrows
            self.results["R2"][curr_indices] = [None] * nrows

        self.cv_idx += 1
        self.idx += nrows

