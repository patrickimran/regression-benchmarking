import numpy as np
from sklearn.model_selection import ParameterGrid


class ParameterGrids:

    def get(algorithm):
        grids = {
            "LinearSVC": {
                'penalty': ['l2'],
                'tol': [1e-4, 1e-3, 1e-2, 1e-1],
                'loss': ['hinge', 'squared_hinge'],
                'random_state': [2018]
            },
            "LinearSVR": {
                "C": [1e-4, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e4],
                "epsilon": [1e-2, 1e-1, 0, 1],
                "loss": ["squared_epsilon_insensitive", "epsilon_insensitive"],
                "random_state": [2018],
            },
            "RidgeClassifier": {
                "alpha": [1e-15, 1e-10, 1e-8, 1e-4],
                "fit_intercept": [True],
                "normalize": [True, False],
                "tol": [1e-1, 1e-2, 1e-3],
                "solver": [
                    "svd",
                    "cholesky",
                    "lsqr",
                    "sparse_cg",
                    "sag",
                    "saga",
                ],
                "random_state": [2018],
            },
            "RidgeRegressor": {
                "alpha": [1e-15, 1e-10, 1e-8, 1e-4],
                "fit_intercept": [True],
                "normalize": [True, False],
                "tol": [1e-1, 1e-2, 1e-3],
                "solver": [
                    "svd",
                    "cholesky",
                    "lsqr",
                    "sparse_cg",
                    "sag",
                    "saga",
                ],
                "random_state": [2018],
            },
            "RandomForestClassifier": {
                "n_estimators": [1000, 5000],
                "criterion": ["gini", "entropy"],
                "max_features": ["sqrt", "log2", None] + list(np.arange(0.2, 1, 0.2)),
                "max_samples": [0.25, 0.5, 0.75, None],
                "max_depth": [None],
                "n_jobs": [-1],
                "random_state": [2020],
                "bootstrap": [True],
                "min_samples_split": list(np.arange(0.2, 1, 0.2)) + [2],
                "min_samples_leaf": list(np.arange(0.01, 0.5, 0.2)) + [1],
            },
            "RandomForestRegressor": {
                'n_estimators': [1000, 5000],
                'criterion': ['mse', 'mae'],
                "max_features": ["sqrt", "log2", None] + list(np.arange(0.2, 1, 0.2)),
                "max_samples": [0.25, 0.5, 0.75, None],
                'max_depth': [None],
                'n_jobs': [-1],
                'random_state': [2020],
                'bootstrap': [True],
                'min_samples_split': list(np.arange(0.2, 1, 0.2)) + [2],
                'min_samples_leaf': list(np.arange(0.01, .5, 0.2)) + [1],
            },
            "GradientBoostingRegressor": {
                'loss': ['ls', 'lad', 'huber', 'quantile'],
                'alpha': [1e-3, 1e-2, 1e-1, 0.5, 0.9],
                'learning_rate': [3e-1, 2e-1, 1e-1, 5e-2],
                'n_estimators': [1000, 5000],
                'criterion': ['friedman_mse', 'mse' 'mae'],
                'max_features': [None, 'sqrt', 'log2', 0.2, 0.4, 0.6, 0.8],
                'max_depth': [None, 10, 100],
                'random_state': [2020],
            },
            "GradientBoostingClassifier": {
                'loss': ['deviance', 'exponential'],
                'alpha': [1e-3, 1e-2, 1e-1, 0.5, 0.9],
                'learning_rate': [3e-1, 2e-1, 1e-1, 5e-2],
                'n_estimators': [1000, 5000],
                'criterion': ['friedman_mse', 'mse' 'mae'],
                'max_features': [None, 'sqrt', 'log2', 0.2, 0.4, 0.6, 0.8],
                'max_depth': [None, 10, 100],
                'random_state': [2020],
            },
            "XGBRegressor": {
                'max_depth': [None, 10, 100],
                'learning_rate': [3e-1, 2e-1, 1e-1, 5e-2],
                'n_estimators': [1000, 5000],
                'objective': ['reg:linear', 'reg:squarederror'],
                'booster': ['gbtree', 'gblinear', 'dart'],
                'gamma': [0, 0.2, 0.5, 1, 3],
                'reg_alpha': [1e-3, 1e-1, 1],
                'reg_lambda': [1e-3, 1e-1, 1],
                'random_state': [2020],
                'silent': [1],
                'n_jobs': [-1],
            },
            "XGBClassifier": {
                'max_depth': [None, 10, 100],
                'learning_rate': [3e-1, 2e-1, 1e-1, 5e-2],
                'n_estimators': [1000, 5000],
                'objective': ['reg:linear', 'reg:squarederror'],
                'booster': ['gbtree', 'gblinear', 'dart'],
                'gamma': [0, 0.2, 0.5, 1, 3],
                'reg_alpha': [1e-3, 1e-1, 1],
                'reg_lambda': [1e-3, 1e-1, 1],
                'random_state': [2020],
                'silent': [1],
                'n_jobs': [-1],
            },
            "ExtraTreesClassifier": {
                "n_estimators": [1000, 5000],
                "criterion": ["gini", "entropy"],
                "max_features": ["sqrt", "log2", None] + list(np.arange(0.2, 1, 0.2)),
                "max_samples": [0.25, 0.5, 0.75, None],
                "max_depth": [None],
                "n_jobs": [-1],
                "random_state": [2020],
                "bootstrap": [True],
                "min_samples_split": list(np.arange(0.2, 1, 0.2)) + [2],
                "min_samples_leaf": list(np.arange(0.01, 0.5, 0.2)) + [1],
            },
            "ExtraTreesRegressor": {
                'n_estimators': [1000, 5000],
                'criterion': ['mse', 'mae'],
                "max_features": ["sqrt", "log2", None] + list(np.arange(0.2, 1, 0.2)),
                "max_samples": [0.25, 0.5, 0.75, None],
                'max_depth': [None],
                'n_jobs': [-1],
                'random_state': [2020],
                'bootstrap': [True],
                'min_samples_split': list(np.arange(0.2, 1, 0.2)) + [2],
                'min_samples_leaf': list(np.arange(0.01, .5, 0.2)) + [1],
            },
            "RadialSVR": {
                'C': [1e-4, 1e-3, 1e-2, 1e-1, 1e1,
                      1e2, 1e3, 1e4, 1e5, 1e6, 1e7],
                'epsilon': [ 1e-5, 1e-4, 1e-3, 1e-2],
                'kernel': ['rbf'],
                'gamma': ['auto', 100, 10, 1, 1e-2, 1e-3,
                          1e-4, 1e-5, 1e-6],
            },
            "RadialSVC": {
                'C': [1e-4, 1e-3, 1e-2, 1e-1, 1e1,
                      1e2, 1e3, 1e4, 1e5, 1e6, 1e7],
                'epsilon': [[ 1e-5, 1e-4, 1e-3, 1e-2],
                'kernel': ['rbf'],
                'gamma': ['auto', 100, 10, 1, 1e-2, 1e-3,
                          1e-4, 1e-5, 1e-6],
            },
            "SigmoidSVR": {
                'C': [1e-4, 1e-3, 1e-2, 1e-1, 1e1,
                      1e2, 1e3, 1e4, 1e5, 1e6, 1e7],
                'epsilon': [ 1e-5, 1e-4, 1e-3, 1e-2],
                'kernel': ['sigmoid'],
                'gamma': ['auto', 100, 10, 1, 1e-2, 1e-3,
                          1e-4, 1e-5, 1e-6],
                'coef0': [0, 1, 10, 100],
            },
            "SigmoidSVC": {
                'C': [1e-4, 1e-3, 1e-2, 1e-1, 1e1,
                      1e2, 1e3, 1e4, 1e5, 1e6, 1e7],
                'epsilon': [ 1e-5, 1e-4, 1e-3, 1e-2],
                'kernel': ['sigmoid'],
                'gamma': ['auto', 100, 10, 1, 1e-2, 1e-3,
                          1e-4, 1e-5, 1e-6],
                'coef0': [0, 1, 10, 100],
            },
        }
        return grids[algorithm]

    def get_reduced(algorithm):
        lr_alphas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.5, 0.9,
                     0.99]
        grids = {
            "ElasticNet": {
                "alpha": lr_alphas,
                "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
                "normalize": [True, False],
                "positive": [True, False],
                "random_state": [2020],
            },
            "KNeighborsRegressor": {
                "n_neighbors": [1, 5, 10, 20, 40],
                "weights": ['uniform', 'distance'],
                # TODO is what about UniFrac, can we use pre-computed
                #  distance matrix?
                # https://docs.scipy.org/doc/scipy/reference/spatial.distance.html # noqa
                "metric": ['braycurtis', 'jaccard']
            },
            "KNeighborsClassifier": {
                "n_neighbors": [1, 5, 10, 20, 40],
                "weights": ['uniform', 'distance'],
                # TODO is what about UniFrac, can we use pre-computed
                #  distance matrix?
                # https://docs.scipy.org/doc/scipy/reference/spatial.distance.html # noqa
                "metric": ['braycurtis', 'jaccard']
            },
            "Lasso": {
                "alpha": lr_alphas,
                "normalize": [True, False],
                "random_state": [2020],
            },
            "LinearRegression": {
                "fit_intercept": [True, False],
                "normalize": [True, False],
            },
            "LogisticRegression": [
                # Lasso
                {
                    'penalty': ['l1'],
                    # these are the only solvers that support l1 penalty
                    'solver': ['saga', 'liblinear'],
                    'C': [1 / alpha for alpha in lr_alphas],
                    'random_state': [2020],
                },
                # ElasticNet
                {
                    'penalty': ['elasticnet'],
                    # this is the only solver that supports elasticnet penalty
                    'solver': ['saga'],
                    'C': [1 / alpha for alpha in lr_alphas],
                    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                    'random_state': [2020],
                },
                # LogisticRegression
                {
                    'penalty': ['none'],
                    'fit_intercept': [True, False],
                    'random_state': [2020],
                },
            ],
            "MLPClassifier": {
                "hidden_layer_sizes": [(50, ), (100, ), (50, 50), (100, 100)],
                "solver": ['adam', 'sgd'],
                "activation": ['tanh', 'relu'],
                "learning_rate": ['constant', 'adaptive'],
                "random_state": [2020],
            },
            "MLPRegressor": {
                "hidden_layer_sizes": [(50, ), (100, ), (50, 50), (100, 100)],
                "solver": ['adam', 'sgd'],
                "activation": ['tanh', 'relu'],
                "learning_rate": ['constant', 'adaptive'],
                "random_state": [2020],
            },
            "RandomForestClassifier": {
                "n_estimators": [10, 100, 1000],
                "criterion": ["gini"],
                "max_features": ["sqrt", "log2", None, 0.4, 0.6],
                "max_samples": [0.25, 0.5, 0.75, None],
                "max_depth": [None, 10, 100],
                "n_jobs": [-1],
                "random_state": [2020],
                "bootstrap": [True],
            },
            "RandomForestRegressor": {
                'n_estimators': [10, 100, 1000],
                'criterion': ['mse'],
                "max_features": ["sqrt", "log2", None, 0.4, 0.6],
                "max_samples": [0.25, 0.5, 0.75, None],
                'max_depth': [None, 10, 100],
                'n_jobs': [-1],
                'random_state': [2020],
                'bootstrap': [True],
            },
            "RidgeClassifier": {
                "alpha": lr_alphas,
                "normalize": [True, False],
                "random_state": [2018],
            },
            "RidgeRegressor": {
                "alpha": lr_alphas,
                "normalize": [True, False],
                "random_state": [2020],
            "GradientBoostingRegressor": {
                'loss': ['ls', 'lad', 'huber', 'quantile'],
                'alpha': [1e-3, 1e-1, 0.5, 0.9],
                'learning_rate': [3e-1, 1e-1, 5e-2],
                'n_estimators': [1000, 5000],
                'criterion': ['mse'],
                'max_features': [None, 'sqrt', 'log2', 0.4, 0.6],
                'max_depth': [None, 10, 100],
                'random_state': [2020],
            },
            "GradientBoostingClassifier": {
                'loss': ['deviance', 'exponential'],
                'alpha': [1e-3, 1e-1, 0.5, 0.9],
                'learning_rate': [3e-1, 1e-1, 5e-2],
                'n_estimators': [1000, 5000],
                'criterion': ['mse'],
                'max_features': [None, 'sqrt', 'log2', 0.4, 0.6],
                'max_depth': [None, 10, 100],
                'random_state': [2020],
            },
            "XGBRegressor": {
                'max_depth': [None, 10, 100],
                'learning_rate': [3e-1, 2e-1, 1e-1, 5e-2],
                'n_estimators': [5000],
                'objective': ['reg:linear'],
                'booster': ['gbtree'],
                'gamma': [0],
                'reg_alpha': [1e-3, 1e-1, 1],
                'reg_lambda': [1e-3, 1e-1, 1],
                'random_state': [2020],
                'silent': [1],
                'n_jobs': [-1],
            },
            "XGBClassifier": {
                'max_depth': [None, 10, 100],
                'learning_rate': [3e-1, 2e-1, 1e-1, 5e-2],
                'n_estimators': [1000, 5000],
                'objective': ['reg:linear'],
                'booster': ['gbtree'],
                'gamma': [0],
                'reg_alpha': [1e-3, 1e-1, 1],
                'reg_lambda': [1e-3, 1e-1, 1],
                'random_state': [2020],
                'silent': [1],
                'n_jobs': [-1],
            },
            "ExtraTreesClassifier": {
                "n_estimators": [5000],
                "criterion": ["gini"],
                "max_features": ["sqrt", "log2", None] + list(np.arange(0.2, 1, 0.2)),
                "max_samples": [0.25, 0.5, 0.75, None],
                "max_depth": [None],
                "n_jobs": [-1],
                "random_state": [2020],
                "bootstrap": [True],
            },
            "ExtraTreesRegressor": {
                'n_estimators': [5000],
                'criterion': ['mse'],
                "max_features": ["sqrt", "log2", None] + list(np.arange(0.2, 1, 0.2)),
                "max_samples": [0.25, 0.5, 0.75, None],
                'max_depth': [None],
                'n_jobs': [-1],
                'random_state': [2020],
                'bootstrap': [True],
            },
            "RadialSVR": {
                'C': [1e-4, 1e-2, 1e-1, 1e1,
                      1e2, 1e4, 1e6,],
                'epsilon': [1e-3, 1e-1],
                'kernel': ['rbf'],
                'gamma': ['auto', 100, 10, 1, 1e-2, 1e-3, 1e-4],
            },
            "RadialSVC": {
                'C': [1e-4, 1e-2, 1e-1, 1e1,
                      1e2, 1e4, 1e6,],
                'epsilon': [1e-3, 1e-1],
                'kernel': ['rbf'],
                'gamma': ['auto', 100, 10, 1, 1e-2, 1e-3, 1e-4],
            },
            "SigmoidSVR": {
                'C': [1e-4, 1e-2, 1e-1, 1e1,
                      1e2, 1e4, 1e6,],
                'epsilon': [1e-3, 1e-1],
                'kernel': ['sigmoid'],
                'gamma': ['auto', 100, 10, 1, 1e-2, 1e-3, 1e-4],
                'coef0': [0, 1, 10, 100],
            },
            "SigmoidSVC": {
                'C': [1e-4, 1e-2, 1e-1, 1e1,
                      1e2, 1e4, 1e6,],
                'epsilon': [1e-3, 1e-1],
                'kernel': ['sigmoid'],
                'gamma': ['auto', 100, 10, 1, 1e-2, 1e-3, 1e-4],
                'coef0': [0, 1, 10, 100],
            },
        }
        return grids[algorithm]

    def get_size(algorithm, reduced=False):
        if reduced:
            grid = ParameterGrids.get_reduced(algorithm)
        else:
            grid = ParameterGrids.get(algorithm)
        size = len(list(ParameterGrid(grid)))
        return size
