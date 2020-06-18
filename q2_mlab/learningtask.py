import json
import biom
import numpy as np
import pandas as pd
import time
from skbio import DistanceMatrix

# CV Methods
from sklearn.model_selection import RepeatedStratifiedKFold

# Metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc, log_loss

# Algorithms
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.svm import SVR, SVC, LinearSVR, LinearSVC

from abc import ABC

# TODO convert to ABC
class LearningTask(ABC):

    algorithms = {}

    def __init__(self, table, metadata, algorithm, params, 
                 distance_matrix):
        self.distance_matrix = distance_matrix
        self.params = json.loads(params)
        self.X = table.transpose().matrix_data
        self.metadata = metadata #.view(pd.Series)
        self.y = self.metadata.to_numpy()
        self.learner = self.algorithms[algorithm]
        self.cv_idx = 0

        # TODO preallocate lists in size n_cv_repeats * len(y) * size(float)
        self.results = {}
        self.results["CV_IDX"] = []
        self.results["SAMPLE_ID"] = []
        self.results["Y_PRED"] = []
        self.results["Y_TRUE"] = []
        self.results["RUNTIME"] = [] 

        # TODO Validate the shapes of X and y, sample_id agreement
        # TODO Validate y is of type int
        # If checks fail, throw exception and end - must be handled in preprocess

    
    def contains_nan(self, y_pred):
        if ((np.any(pd.isnull(y_pred))) or
        (not np.all(np.isfinite(y_pred)))):
            return True
        else:
            return False
    
    def tabularize(self):
        results_table = pd.DataFrame.from_dict(self.results)
        return results_table

    def print_attributes(self): # TODO delete
        print(self.learner)
        print(self.params)


class ClassificationTask(LearningTask):

    algorithms = {
        "KNeighborsClassifier": KNeighborsClassifier,
        "RandomForestClassifier": RandomForestClassifier,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "XGBClassifier": XGBClassifier,
        "LinearSVC": LinearSVC,
    }

    def __init__(self, table, metadata, algorithm, params, 
                 distance_matrix):
        super().__init__(table, metadata, algorithm, params, 
                         distance_matrix)
        
        kfold = RepeatedStratifiedKFold(5, 3, random_state=2020)
        self.splits = kfold.split(self.X, self.y)

        self.results["Y_PROB"] = []
        self.results["PRECISIONS"] = []
        self.results["ACCURACY"] = []
        self.results["AUPRC"] = []
        self.results["AUROC"] = []
    
    def cv_fold(self, train_index,test_index):
        X_train, X_test = self.X[train_index], self.X[test_index]
        y_train, y_test = self.y[train_index], self.y[test_index]
        y_test_ids = self.metadata.index[test_index]

        # Start timing
        start = time.process_time()
        m = self.learner(**self.params)
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        probas = m.predict_proba(X_test) # Classification only
        # End timimg
        end = time.process_time()

        # Record 
        runtime = end-start
        nrows = len(y_pred)
        self.results["RUNTIME"].extend([runtime] * nrows)
        self.results["CV_IDX"].extend([self.cv_idx] * nrows)
        self.results["Y_PRED"].extend(y_pred)
        self.results["Y_TRUE"].extend(y_test)
        self.results["Y_PROB"].extend(probas)
        self.results["SAMPLE_ID"].extend(y_test_ids)

       
        mean_recall = np.linspace(0, 1, 100)
        if not self.contains_nan(y_pred):
            probas = probas[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, probas)
            self.results["PRECISIONS"].extend(
                [np.interp(mean_recall, recall[::-1], precision[::-1])] * nrows
            )
            self.results["AUPRC"].extend([auc(recall, precision)] * nrows)
            self.results["AUROC"].extend([roc_auc_score(y_test, 
                                         probas)] * nrows)
            acc_score = accuracy_score(y_pred, y_test)
            self.results["ACCURACY"].extend([acc_score] * nrows)
        else:
            self.results["AUPRC"].extend([None] * nrows) 
            self.results["AUROC"].extend([None] * nrows)
            self.results["ACCURACY"].extend([None] * nrows)
        
        self.cv_idx += 1


class RegressionTask(LearningTask):

    algorithms = {
        "KNeighborsRegressor": KNeighborsRegressor,
        "RandomForestRegressor": RandomForestRegressor,
        "GradientBoostingRegressor": GradientBoostingRegressor,
        "XGBRegressor": XGBRegressor,
        "LinearSVR": LinearSVR,
    }

    def __init__(self, table, metadata, algorithm, params, 
                 distance_matrix):
        super().__init__(table, metadata, algorithm, params, 
                         distance_matrix)
        
        # TODO replace kfold with repeated SORTED stratified
        kfold = RepeatedStratifiedKFold(5, 3, random_state=2020)
        self.splits = kfold.split(self.X, self.y)

        self.results["RMSE"] = []
        self.results["R2"] = []

    def cv_fold(self, train_index,test_index):
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

        # Record 
        runtime = end-start
        nrows = len(y_pred)
        self.results["RUNTIME"].extend([runtime] * nrows)
        self.results["CV_IDX"].extend([self.cv_idx] * nrows)
        self.results["Y_PRED"].extend(y_pred)
        self.results["Y_TRUE"].extend(y_test)
        self.results["SAMPLE_ID"].extend(y_test_ids)

        if not self.contains_nan(y_pred):
            rmse = mean_squared_error(y_test, y_pred)
            self.results["RMSE"].extend([rmse] * nrows)
            r2 = r2_score(y_test, y_pred)
            self.results["R2"].extend([r2] * nrows) 
        else:
            self.results["RMSE"].extend([None] * nrows) 
            self.results["R2"].extend([None] * nrows)
        
        self.cv_idx += 1