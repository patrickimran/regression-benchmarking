# ----------------------------------------------------------------------------
# Copyright (c) 2016-2020, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import os
import tempfile
import unittest

import pandas as pd
import numpy as np
import qiime2
import skbio

#from q2_emperor import plot, procrustes_plot, biplot, generic_plot


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        

        self.metadata = qiime2.Metadata(
            pd.DataFrame({'val1': ['1.0', '2.0', '3.0', '4.0'],
                          'val2': ['3.3', '3.5', '3.6', '3.9']},
                         index=pd.Index(['A', 'B', 'C', 'D'], name='id')))

    def test_plot(self):
        with tempfile.TemporaryDirectory() as output_dir:
            plot(output_dir, self.pcoa, self.metadata)
            index_fp = os.path.join(output_dir, 'index.html')
            self.assertTrue(os.path.exists(index_fp))
            self.assertTrue('src="./emperor.html"' in open(index_fp).read())

    