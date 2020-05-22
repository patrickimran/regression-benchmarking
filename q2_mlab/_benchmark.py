import numpy as np
import pandas as pd
import biom
import time
import ast
from skbio import DistanceMatrix

# CV Methods
from sklearn.model_selection import RepeatedStratifiedKFold

# Metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

# Algorithms
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier


def benchmark_classify(ctx, table, metadata, classifier, params, n_jobs,
                       distance_matrix=None,):

    classifiers = {
        "KNeighborsClassifier": KNeighborsClassifier,  # TODO metric names
        "RandomForestClassifier": RandomForestClassifier,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "XGBClassifier": XGBClassifier
    }
    clf = classifiers[classifier]

    columns = ['SAMPLE_ID', 'Y_TRUE', 'Y_PRED', 'CV_IDX', 'ACCURACY', 'AUROC',
               'AUPRC', 'RUNTIME']
    results_table = pd.DataFrame(columns=columns)
    param_dict = ast.literal_eval(params)

    X = table.view(biom.Table).matrix_data
    y = metadata.view(pd.Series).values

    # Toggle between multiclass or binary classification
    _multiclass = False
    unique_classes = metadata.view(pd.Series).unique().values
    if len(unique_classes) > 2:
        _multiclass = True

    # TODO Validate the shapes of X and y, sample_id agreement
    splits = RepeatedStratifiedKFold(5, 3, random_state=2020).split(X, y)
    cv_idx = 0
    CV_IDX = []
    Y_PRED = []
    Y_TRUE = []
    SAMPLE_ID = []

    # Start Timing
    start = time.process_time()
    for train_index, test_index in splits:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        y_train = np.asarray(y_train, dtype='int')
        y_test_ids = y.index[test_index]

        m = clf(**param_dict)

        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)

        CV_IDX.extend([cv_idx] * len(y_pred))
        Y_PRED.extend(y_pred)
        Y_TRUE.extend(y_test)
        SAMPLE_ID.extend(y_test_ids)

        cv_idx += 1
    # End Timing
    end = time.process_time()
    param_runtime = end-start
    results_table['RUNTIME'] = [param_runtime] * results_table.shape[0]
    results_table['CV_IDX'] = CV_IDX
    results_table['Y_TRUE'] = Y_TRUE
    results_table['Y_PRED'] = Y_PRED
    results_table['SAMPLE_ID'] = SAMPLE_ID

    # Check for NaN in predictions
    _contains_nan = False
    predictions = results_table['Y_PRED'].values
    if ((np.any(pd.isnull(predictions))) or
       (not np.all(np.isfinite(predictions)))):
        _contains_nan = True

    # Calculate metrics for each fold in this set
    fold_acc = pd.DataFrame(index=list(range(0, cv_idx)))
    groups = results_table.groupby('CV_IDX')
    if _contains_nan:
        fold_acc['ACCURACY'] = groups.apply(lambda x:
                                            np.sqrt(x['Y_TRUE'].values))
        fold_acc['ACCURACY'] = [None] * fold_acc.shape[0]
        fold_acc['AUROC'] = [None] * fold_acc.shape[0]
    else:
        fold_acc['ACCURACY'] = groups.apply(lambda x:
                                            np.sqrt(accuracy_score(
                                                    x['Y_PRED'].values,
                                                    x['Y_TRUE'].values)))

    # TODO Add AUROC, AUPRC
    if _multiclass:
        pass
        # AUROC
    else:
        pass
        # AUPRC

    # TODO Test this merge of per-fold results onto results_table
    results_table = pd.merge(fold_acc, results_table, on='CV_IDX', how='outer')

    # Transform into the new Results Table semantic type
    results = ctx.make_artifact("SampleData[Results]", results_table)
    return results


def benchmark_regress(ctx, table, metadata, regressor, distance_matrix,
                      params, n_jobs):

    regressors = {
        "KNeighborsRegressor": KNeighborsRegressor,  # TODO metric names
        "RandomForestRegressor": RandomForestRegressor,
        "GradientBoostingRegressor": GradientBoostingRegressor,
        "XGBRegressor": XGBRegressor
    }
    reg = regressors[regressor]

    columns = ['SAMPLE_ID', 'Y_TRUE', 'Y_PRED', 'CV_IDX', 'ACCURACY', 'AUROC',
               'AUPRC', 'RUNTIME']
    results_table = pd.DataFrame(columns=columns)
    param_dict = ast.literal_eval(params)

    X = table.view(biom.Table).matrix_data
    y = metadata.view(pd.Series).values

    # TODO Validate the shapes of X and y, sample_id agreement
    # TODO Repeated *Sorted* StratifiedKFold
    splits = RepeatedStratifiedKFold(5, 3, random_state=2020).split(X, y)
    cv_idx = 0
    CV_IDX = []
    Y_PRED = []
    Y_TRUE = []
    SAMPLE_ID = []

    # Start Timing
    start = time.process_time()
    for train_index, test_index in splits:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        y_train = np.asarray(y_train, dtype='int')
        y_test_ids = y.index[test_index]

        m = reg(**param_dict)

        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)

        CV_IDX.extend([cv_idx] * len(y_pred))
        Y_PRED.extend(y_pred)
        Y_TRUE.extend(y_test)
        SAMPLE_ID.extend(y_test_ids)

        cv_idx += 1

    end = time.process_time()
    param_runtime = end-start
    results_table['RUNTIME'] = [param_runtime] * results_table.shape[0]
    results_table['CV_IDX'] = CV_IDX
    results_table['Y_TRUE'] = Y_TRUE
    results_table['Y_PRED'] = Y_PRED
    results_table['SAMPLE_ID'] = SAMPLE_ID

    # Check for NaN in predictions
    contains_nan = False
    predictions = results_table['Y_PRED'].values
    if ((np.any(pd.isnull(predictions))) or
       (not np.all(np.isfinite(predictions)))):
        contains_nan = True

    # Calculate metrics for each fold in this set
    fold_rmse = pd.DataFrame()
    groups = results_table.groupby('CV_IDX')
    if contains_nan:
        fold_rmse['RMSE'] = groups.apply(lambda x: np.sqrt(x['Y_TRUE'].values))
        fold_rmse['RMSE'] = [None] * fold_rmse.shape[0]
    else:
        fold_rmse['RMSE'] = groups.apply(lambda x:
                                         np.sqrt(mean_squared_error(
                                                 x['Y_PRED'].values,
                                                 x['Y_TRUE'].values)))

    # TODO Add R^2

    # TODO Test this merge of per-fold results onto results_table
    results_table = pd.merge(fold_rmse, results_table, on='CV_IDX',
                             how='outer')

    # Transform into the new Results Table semantic type
    results = ctx.make_artifact("SampleData[Results]", results_table)
    return results
