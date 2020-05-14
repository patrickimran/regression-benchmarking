from os import path
import tempfile

from qiime2.plugin.testing import TestPluginBase

import pandas as pd
from qiime2 import Artifact, Metadata

from q2_mlab._preprocess import clean_metadata


class PreprocessingTests(TestPluginBase):
    package = 'q2_mlab'

    def setUp(self):
        super().setUp()
        self.preprocess = self.plugin.pipelines['preprocess']

        continuous_metadata = pd.DataFrame({'val1': ['1.0', '2.0',
                                                     '3.0', '4.0'],
                                            'contains_nan': ['3.3', '3.5',
                                                             None, '3.9']},
                                           index=pd.Index(['A', 'B', 'C', 'D'],
                                           name='id'))
        self.continuous_metadata = continuous_metadata

        discrete_metadata = pd.DataFrame({'contains_nan': ['oral', 'gut',
                                                           None, 'skin'],
                                          'body_site': ['gut', 'skin',
                                                        'oral', 'gut']},
                                         index=pd.Index(['A', 'B', 'C', 'D'],
                                         name='id'))
        self.discrete_metadata = discrete_metadata

        self.mp_sample_metadata = Metadata.load('data/sample-metadata.tsv')
        self.mp_table = Artifact.load('data/table.qza')
        self.mp_rooted_tree = Artifact.load('data/rooted-tree.qza')
        self.mp_unrooted_tree = Artifact.load('data/unrooted-tree.qza')

    def test_preprocess_output(self):

        results = self.preprocess(table=self.mp_table,
                                  metadata=self.mp_sample_metadata,
                                  phylogeny=self.mp_rooted_tree,
                                  sampling_depth=1000,
                                  min_frequency=10,
                                  target_variable="body-site",
                                  discrete=True,
                                  with_replacement=False,
                                  n_jobs=1)
        self.assertEqual(len(results), 8)

    def test_clean_metadata(self):
        clean_df = clean_metadata(self.continuous_metadata,
                                  target_variable='val1',
                                  discrete=False)
        self.assertTupleEqual(clean_df.shape, (4, 1))
        self.assertIsInstance(clean_df['val1'][0], (int, float))

        clean_df2 = clean_metadata(self.continuous_metadata,
                                   target_variable='contains_nan',
                                   discrete=False)
        self.assertTupleEqual(clean_df2.shape, (3, 1))

        clean_df3 = clean_metadata(self.discrete_metadata,
                                   target_variable='contains_nan',
                                   discrete=True)
        self.assertTupleEqual(clean_df3.shape, (3, 1))
