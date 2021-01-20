import unittest
import numpy as np
import enum
import tqdm
import random
import math
import os
import biom
from itertools import cycle
from q2_mlab.learningtask import RegressionTask, ClassificationTask
from q2_mlab._parameters import ParameterGrids
from sklearn.model_selection import ParameterGrid


class Task(enum.Enum):
    REGRESSION = 1
    CLASSIFICATION = 2


def create_synthetic_data(
    n_samples, n_features, task=Task.REGRESSION, count=500
):
    X = np.random.multinomial(
        count, [1 / n_features] * n_features, size=n_samples
    )
    # make X sparse (randomly set half of entries to 0)
    num_zero = int(X.size / 0.5)
    X0_indices = np.random.randint(0, high=n_samples, size=num_zero)
    X1_indices = np.random.randint(0, high=n_features, size=num_zero)
    X[X0_indices, X1_indices] = 0
    # sum normalize X
    X = X / X.sum(axis=1, keepdims=True)
    # clear nans that come from division of all 0 rows
    is_nan = np.isnan(X)
    X[is_nan] = 0

    # Convert to biom table and return its sparse matrix representation
    table = biom.Table(
        data=X,
        observation_ids=list(range(0, X.shape[0])),
        sample_ids=list(range(0, X.shape[1]))
    )
    X = table.matrix_data

    if task == Task.REGRESSION:
        # We can assume non-negative target variables, a requirement for
        # 'poisson' loss in HistGradientBoostingRegressor
        y = np.absolute(np.random.normal(size=n_samples))
    else:
        assert task == Task.CLASSIFICATION
        # just alternate y labels so we can make sure we get at least one
        # sample from each class
        y = [y for y, _ in zip(cycle([0, 1]), range(n_samples))]

    return X, y


class ParameterGridsTests(unittest.TestCase):

    regression_algorithms = [
        # Ensemble
        "RandomForestRegressor",
        "ExtraTreesRegressor",
        # Boosting
        "GradientBoostingRegressor",
        "XGBRegressor",
        "LGBMRegressor_RF",
        "LGBMRegressor_GBDT",
        "HistGradientBoostingRegressor",
        "AdaBoostRegressor",
        # Linear/Logistic Reg
        "RidgeRegressor",
        "ElasticNet",
        "Lasso",
        "LinearRegression",
        # SVM
        "LinearSVR",
        "RadialSVR",
        "SigmoidSVR",
        # Other
        "KNeighborsRegressor",
        "MLPRegressor",
    ]
    classification_algorithms = [
        # Ensemble
        "RandomForestClassifier",
        "ExtraTreesClassifier",
        # Boosting
        "GradientBoostingClassifier",
        "XGBClassifier",
        "LGBMClassifier_RF",
        "LGBMClassifier_GBDT",
        "HistGradientBoostingClassifier",
        "AdaBoostClassifier",
        # Linear/Logistic Reg
        "RidgeClassifier",
        "LogisticRegression_ElasticNet",
        "LogisticRegression_Lasso",
        "LogisticRegression",
        # SVM
        "LinearSVC",
        "RadialSVC",
        "SigmoidSVC",
        # Other
        "KNeighborsClassifier",
        "MLPClassifier",
    ]

    def setUp(self):
        self.reduced_hyperparameter_space = os.environ.get("TRAVIS") == 'true'
        self.reduced_size = 0.1  # % to reduce parameter spaces by for Travis
        self.X_reg, self.y_reg = create_synthetic_data(
            n_samples=16, n_features=4, task=Task.REGRESSION,
        )
        self.X_class, self.y_class = create_synthetic_data(
            n_samples=16, n_features=4, task=Task.CLASSIFICATION,
        )

    # def test_coverage(self):
    #     self.assertCountEqual(
    #         list(ParameterGrids.full_grids.keys()),
    #         self.classification_algorithms+self.regression_algorithms
    #     )

    def test_regression_param_grids(self, algorithms=None):
        if algorithms is None:
            algorithms = self.regression_algorithms
        task = RegressionTask
        self._test_grids(
            self.X_reg, self.y_reg, algorithms, task,
        )

    def test_classification_param_grids(self, algorithms=None):
        if algorithms is None:
            algorithms = self.classification_algorithms
        task = ClassificationTask
        self._test_grids(
            self.X_class, self.y_class, algorithms, task,
        )

    def _test_grids(self, X, y, algorithms, task):
        total_its = sum(
                ParameterGrids.get_size(alg) for alg in algorithms
        )
        if self.reduced_hyperparameter_space:
            total_its = sum(
                math.ceil(
                    self.reduced_size*ParameterGrids.get_size(alg)
                ) for alg in algorithms
            )
        with tqdm.tqdm(total=total_its) as pbar:
            for alg in algorithms:
                
                # Convert to dense array for HistGradientBoosting
                dense_only_algorithms = {
                    "HistGradientBoostingClassifier",
                    "HistGradientBoostingRegressor"
                }
                if alg in dense_only_algorithms:
                    X = X.todense()

                estimator_cls = task.algorithms[alg]
                grid = ParameterGrids.get(alg)
                param_list = list(ParameterGrid(grid))
                if self.reduced_hyperparameter_space:
                    param_list = random.sample(
                        param_list,
                        math.ceil(self.reduced_size*len(grid))
                    )
                for params in param_list:
                    with self.subTest(algorithm=alg, params=params):
                        estimator = estimator_cls(**params)
                        estimator.fit(X, y)
                        pbar.update(1)


if __name__ == "__main__":
    unittest.main()
