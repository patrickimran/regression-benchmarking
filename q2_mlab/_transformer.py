# ----------------------------------------------------------------------------
# Copyright (c) 2016-2020, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from . import ResultsDirectoryFormat, ResultsFormat
from qiime2 import Metadata
import pandas as pd
from .plugin_setup import plugin


@plugin.register_transformer
def _1(data: Metadata) -> ResultsFormat:
    ff = ResultsFormat()
    with ff.open() as fh:
        df = data.to_dataframe()
        df.to_csv(fh, sep='\t', header=True)
    return ff

@plugin.register_transformer
def _2(data: pd.DataFrame) -> ResultsFormat:
    ff = ResultsFormat()
    with ff.open() as fh:
        data.to_csv(fh, sep='\t', header=True)
    return ff

