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
            "C": [1e-4, 1e-2, 1e-1, 1e1, 1e2, 1e4],
            "epsilon": [1e-2, 1e-1, 0, 1],
            "loss": ["squared_epsilon_insensitive", "epsilon_insensitive"],
            "random_state": [2018],
            "max_iter": [20000],
        }

        LinearSVC_grids = {
            "penalty": ["l2"],
            "tol": [1e-4, 1e-2, 1e-1],
            "random_state": [2018],
            "max_iter": [20000],
        }

        self.svr_params = json.dumps(list(ParameterGrid(LinearSVR_grids))[0])
        self.svc_params = json.dumps(list(ParameterGrid(LinearSVC_grids))[0])
        self.default_params = "{}"

    def testRegressionTask(self):

        (results,) = self.unit_benchmark(
            table=self.table,
            metadata=self.metadata,
            algorithm="LinearSVR",
            params=self.svr_params,
            n_repeats=self.n_repeats,
            distance_matrix=self.distance_matrix,
        )

        table_df = self.table.view(pd.DataFrame)
        results_df = results.view(pd.DataFrame)

        # Assert format and content of results
        expected_shape = (table_df.shape[0] * self.n_repeats, 7)
        self.assertTupleEqual(results_df.shape, expected_shape)

    def testClassificationNoProba(self):
        (results,) = self.unit_benchmark(
            table=self.table,
            metadata=self.metadata,
            algorithm="LinearSVC",
            params=self.svc_params,
            n_repeats=self.n_repeats,
            distance_matrix=self.distance_matrix,
        )

        table_df = self.table.view(pd.DataFrame)
        results_df = results.view(pd.DataFrame)

        # Assert format and content of results
        expected_shape = (table_df.shape[0] * self.n_repeats, 10)
        self.assertTupleEqual(results_df.shape, expected_shape)

        # Assert columns exist and are null
        expected_cols = [
            "CV_IDX",
            "SAMPLE_ID",
            "Y_PRED",
            "Y_TRUE",
            "RUNTIME",
            "Y_PROB",
            "ACCURACY",
            "AUPRC",
            "AUROC",
            "F1",
        ]
        self.assertListEqual(list(results_df.columns), expected_cols)

    def testClassificationWithProba(self):
        (results,) = self.unit_benchmark(
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

        expected_cols = [
            "CV_IDX",
            "SAMPLE_ID",
            "Y_PRED",
            "Y_TRUE",
            "RUNTIME",
            "Y_PROB",
            "ACCURACY",
            "AUPRC",
            "AUROC",
            "F1",
        ]
        self.assertListEqual(list(results_df.columns), expected_cols)
