import pandas as pd
import json
from os import path
from qiime2 import Artifact
from qiime2.plugin.testing import TestPluginBase
from sklearn.model_selection import ParameterGrid


class UnitBenchmarkTests(TestPluginBase):
    package = "q2_mlab"

    def setUp(self):
        super().setUp()
        self.unit_benchmark = self.plugin.methods["unit_benchmark"]

        TEST_DIR = path.split(__file__)[0]
        table = path.join(
            TEST_DIR, "movingpicturestest/filtered_rarefied_table.qza"
        )
        metadata = path.join(
            TEST_DIR, "movingpicturestest/filtered_metadata.qza"
        )
        distance_matrix = path.join(
            TEST_DIR, "movingpicturestest/aitchison_distance_matrix.qza"
        )

        self.table = Artifact.load(table)
        self.metadata = Artifact.load(metadata)
        self.distance_matrix = Artifact.load(distance_matrix)
        self.n_repeats = 1

        LinearSVR_grids = {
            "C": [1e-4, 1e-3, 1e-2, 1e-1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7],
            "epsilon": [1e-2, 1e-1, 0, 1],
            "loss": ["squared_epsilon_insensitive", "epsilon_insensitive"],
            "random_state": [2018],
        }

        LinearSVC_grids = {
            "penalty": {"l1", "l2"},
            "tol": [1e-4, 1e-2, 1e-1],
            "loss": ["hinge", "squared_hinge"],
            "random_state": [2018],
        }

        self.reg_params = json.dumps(list(ParameterGrid(LinearSVR_grids))[0])
        self.clf_params = json.dumps(list(ParameterGrid(LinearSVC_grids))[0])
        self.default_params = "{}"

    def testRegressionTask(self):

        results, = self.unit_benchmark(
            table=self.table,
            metadata=self.metadata,
            algorithm="LinearSVR",
            params=self.default_params,
            n_repeats=self.n_repeats,
            distance_matrix=self.distance_matrix,
        )

        table_df = self.table.view(pd.DataFrame)
        results_df = results.view(pd.DataFrame)

        # Assert format and content of results
        expected_shape = (table_df.shape[0] * self.n_repeats, 7)
        self.assertTupleEqual(results_df.shape, expected_shape)

    def testClassificationNoProba(self):
        results, = self.unit_benchmark(
            table=self.table,
            metadata=self.metadata,
            algorithm="LinearSVC",
            params=self.default_params,
            n_repeats=self.n_repeats,
            distance_matrix=self.distance_matrix,
        )

        table_df = self.table.view(pd.DataFrame)
        results_df = results.view(pd.DataFrame)

        # Assert format and content of results
        expected_shape = (table_df.shape[0] * self.n_repeats, 10)
        self.assertTupleEqual(results_df.shape, expected_shape)

    def testClassificationWithProba(self):
        results, = self.unit_benchmark(
            table=self.table,
            metadata=self.metadata,
            algorithm="RandomForestClassifier",
            params=self.default_params,
            n_repeats=self.n_repeats,
            distance_matrix=self.distance_matrix,
        )

        table_df = self.table.view(pd.DataFrame)
        results_df = results.view(pd.DataFrame)

        # Assert format and content of results
        expected_shape = (table_df.shape[0] * self.n_repeats, 10)
        self.assertTupleEqual(results_df.shape, expected_shape)

    def testContainsNaN(self):
        pass

    def testTabularize(self):
        pass
