import pandas as pd
from qiime2 import Artifact, Metadata
from os import path
from q2_mlab._preprocess import clean_metadata
from qiime2.plugin.testing import TestPluginBase


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

        TEST_DIR = path.split(__file__)[0]
        sample_metadata_path = path.join(TEST_DIR, 'data/sample-metadata.tsv')
        table_path = path.join(TEST_DIR, 'data/table.qza')
        rooted_tree_path = path.join(TEST_DIR, 'data/rooted-tree.qza')
        unrooted_tree_path = path.join(TEST_DIR, 'data/unrooted-tree.qza')

        self.mp_sample_metadata = Metadata.load(sample_metadata_path)
        self.mp_table = Artifact.load(table_path)
        self.mp_rooted_tree = Artifact.load(rooted_tree_path)
        self.mp_unrooted_tree = Artifact.load(unrooted_tree_path)

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
