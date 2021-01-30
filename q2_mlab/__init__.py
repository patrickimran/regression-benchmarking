# ----------------------------------------------------------------------------
# Copyright (c) 2020, mlab development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from ._preprocess import preprocess
from ._benchmark import unit_benchmark, _unit_benchmark
from ._version import get_versions
from .learningtask import LearningTask, ClassificationTask, RegressionTask
from ._type import Target, Results
from ._format import ResultsDirectoryFormat, ResultsFormat
from ._parameters import ParameterGrids
from .orchestrator import orchestrate_hyperparameter_search
from .doctor import (
    doctor_hyperparameter_search,
    sort_result_artifact_filenames,
    parse_info,
    get_results,
    filter_duplicate_parameter_results,
)
from .collector import collect_hyperparameter_search

__version__ = get_versions()["version"]

del get_versions

__all__ = [
    "preprocess",
    "unit_benchmark",
    "_unit_benchmark",
    "orchestrate_hyperparameter_search",
    "collect_hyperparameter_search",
    "doctor_hyperparameter_search",
    "sort_result_artifact_filenames",
    "parse_info",
    "get_results",
    "filter_duplicate_parameter_results",
    "ResultsDirectoryFormat",
    "ResultsFormat",
    "Target",
    "Results",
    "LearningTask",
    "ClassificationTask",
    "RegressionTask",
    "ParameterGrids",
]
