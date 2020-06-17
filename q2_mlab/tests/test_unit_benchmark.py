import pandas as pd
import json
from os import path
from qiime2 import Artifact, Metadata
from q2_mlab._preprocess import clean_metadata
from qiime2.plugin.testing import TestPluginBase
from pandas.testing import assert_series_equal
from sklearn.model_selection import ParameterGrid

class UnitBenchmarkTests(TestPluginBase):
    package = 'q2_mlab'

    def setUp(self):
        super().setUp()
        self.unit_benchmark = self.plugin.methods['unit_benchmark']

        TEST_DIR = path.split(__file__)[0]
        table = path.join(TEST_DIR, 
            'movingpicturestest/filtered_rarefied_table.qza')
        metadata = path.join(TEST_DIR,
            'movingpicturestest/filtered_metadata.qza')
        distance_matrix = path.join(TEST_DIR, 
            'movingpicturestest/aitchison_distance_matrix.qza')

        self.table = Artifact.load(table)
        self.metadata = Artifact.load(metadata)
        self.distance_matrix = Artifact.load(distance_matrix)
        
        LinearSVR_grids = {'C': [1e-4, 1e-3, 1e-2, 1e-1, 1e1, 
                                 1e2, 1e3, 1e4, 1e5, 1e6, 1e7],
                           'epsilon':[1e-2, 1e-1, 0, 1],
                           'loss': ['squared_epsilon_insensitive', 
                                    'epsilon_insensitive'],
                           'random_state': [2018]
        }

        LinearSVC_grids = {'penalty': {'l1', 'l2'},
                            'tol':[1e-4, 1e-2, 1e-1],
                            'loss': ['hinge', 'squared_hinge'],
                            'random_state': [2018]
        }

        self.reg_params = json.dumps(list(ParameterGrid(LinearSVR_grids))[0])
        self.clf_params = json.dumps(list(ParameterGrid(LinearSVC_grids))[0])

    
    def testRegressionTask(self):

        results = self.unit_benchmark(table = self.table,
                                      metadata = self.metadata,
                                      algorithm = "LinearSVR",
                                      params = self.reg_params,
                                      n_jobs = 1,
                                      distance_matrix = self.distance_matrix)
        
        print(results)
        # Assert format and content of results

    def testClassificationTaskInit(self):
        pass

    def testContainsNaN(self):
        pass

    def testTabularize(self):
        pass
'''
    def testRegressionTaskInit(self):
        worker = RegressionTask(table = self.table,
                                metadata = self.metadata,
                                algorithm = "LinearSVR",
                                params = self.reg_params, 
                                distance_matrix = self.distance_matrix)
        
        # Assert worker's properties are init correctly
'''



    
    
        
        
        

        