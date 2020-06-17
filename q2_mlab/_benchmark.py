# Import learning classes
import pandas as pd
import biom
from qiime2 import Metadata
from skbio.stats.distance import DistanceMatrix
from .learningtask import RegressionTask, ClassificationTask

def unit_benchmark(table: biom.Table, 
                   metadata: pd.Series,
                   algorithm: str, 
                   params: str, 
                   n_jobs: int,
                   distance_matrix: DistanceMatrix) -> pd.DataFrame:
    # TODO: Note that n_jobs is not used

    if algorithm in RegressionTask.algorithms:
        worker = RegressionTask(table, metadata, algorithm, params, 
                              distance_matrix)
    elif algorithm in ClassificationTask.algorithms:
        worker = ClassificationTask(table, metadata, algorithm, params, 
                              distance_matrix)
    else:
        # TODO more specific error message
        # PSL exceptions or a new subclass
        raise Exception(algorithm + " is not an accepted algorithm") 

    for train_index, test_index in worker.splits:
        worker.cv_fold(train_index, test_index)

    results_table = worker.tabularize()

    # Transform into the new Results Table semantic type
    # results = ctx.make_artifact("SampleData[Results]", results_table)

    return results_table #results
