import unittest
import numpy as np
import enum
import tqdm
from itertools import cycle
from q2_mlab.learningtask import RegressionTask, ClassificationTask
from q2_mlab._parameters import ParameterGrids
from sklearn.model_selection import ParameterGrid


class Task(enum.Enum):
    REGRESSION = 1
    CLASSIFICATION = 2


def create_synthetic_data(n_samples, n_features, task=Task.REGRESSION,
                          count=500):
    X = np.random.multinomial(count, [1 / n_features] * n_features,
                              size=n_samples)
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

    if task == Task.REGRESSION:
        y = np.random.normal(size=n_samples)
    else:
        assert task == Task.CLASSIFICATION
        # just alternate y labels so we can make sure we get at least one
        # sample from each class
        y = [y for y, _ in zip(cycle([0, 1]), range(n_samples))]

    return X, y


class ParameterGridsTests(unittest.TestCase):

    regression_algorithms = [
        "ElasticNet",
        "KNeighborsRegressor",
        "Lasso",
        "LinearRegression",
        "MLPRegressor",
        "RandomForestRegressor",
        "RidgeRegressor",
    ]
    classification_algorithms = [
        "KNeighborsClassifier",
        "LogisticRegression",
        "MLPClassifier",
        "RandomForestClassifier",
        "RidgeClassifier",
    ]

    def setUp(self):
        self.X_reg, self.y_reg = create_synthetic_data(
            n_samples=3,
            n_features=4,
            task=Task.REGRESSION,
        )
        self.X_class, self.y_class = create_synthetic_data(
            n_samples=3,
            n_features=4,
            task=Task.CLASSIFICATION,
        )

    def test_reduced_regression_param_grids(self, algorithms=None):
        if algorithms is None:
            algorithms = self.regression_algorithms
        task = RegressionTask
        self._test_grids(self.X_reg, self.y_reg,
                         algorithms, task, reduced=True)

    def test_reduced_classification_param_grids(self, algorithms=None):
        if algorithms is None:
            algorithms = self.classification_algorithms
        task = ClassificationTask
        self._test_grids(self.X_class, self.y_class,
                         algorithms, task, reduced=True)

    def _test_grids(self, X, y, algorithms, task, reduced):
        total_its = sum(ParameterGrids.get_size(alg, reduced=reduced) for alg
                        in algorithms)
        with tqdm.tqdm(total=total_its) as pbar:
            for alg in algorithms:
                estimator_cls = task.algorithms[alg]
                param_grid = ParameterGrids.get_reduced(alg)
                for params in list(ParameterGrid(param_grid)):
                    with self.subTest(algorithm=alg, params=params):
                        estimator = estimator_cls(**params)
                        estimator.fit(X, y)
                        pbar.update(1)


if __name__ == '__main__':
    unittest.main()
