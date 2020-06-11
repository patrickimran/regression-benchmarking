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
from sklearn.metrics import roc_curve, auc

# Import learning classes
from .learningtask import RegressionTask, ClassificationTask

def unit_benchmark(ctx, table, metadata, algorithm, params, n_jobs,
                       distance_matrix=None,):
    # TODO: Note that n_jobs is not used
    if algorithm in RegressionTask.algorithms:
        worker = RegressionTask(table, metadata, algorithm, params, 
                              distance_matrix)
    elif algorithm in ClassificationTask.algorithms:
        worker = ClassificationTask(table, metadata, algorithm, params, 
                              distance_matrix)
    else:
        # TODO more specific error message
        raise Exception(algorithm + " is not an accepted algorithm") 

    for train_index, test_index in worker.splits:
        worker.cv_fold(train_index, test_index)

    results_table = worker.tabularize()

    # Transform into the new Results Table semantic type
    results = ctx.make_artifact("SampleData[Results]", results_table)
    return results
