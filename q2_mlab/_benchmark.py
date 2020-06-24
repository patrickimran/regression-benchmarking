# Import learning classes
import pandas as pd
import biom
from qiime2 import Metadata
from skbio.stats.distance import DistanceMatrix
from .learningtask import RegressionTask, ClassificationTask


def unit_benchmark(
    table: biom.Table,
    metadata: pd.Series,
    algorithm: str,
    params: str,
    distance_matrix: DistanceMatrix,
    n_repeats: int = 3,
) -> pd.DataFrame:

    if algorithm in RegressionTask.algorithms:
        worker = RegressionTask(
            table, metadata, algorithm, params, n_repeats, distance_matrix
        )
    elif algorithm in ClassificationTask.algorithms:
        worker = ClassificationTask(
            table, metadata, algorithm, params, n_repeats, distance_matrix
        )
    else:
        raise ValueError(algorithm + " is not an accepted algorithm")

    for train_index, test_index in worker.splits:
        worker.cv_fold(train_index, test_index)

    results_table = worker.tabularize()

    return results_table
