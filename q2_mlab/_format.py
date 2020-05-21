# ----------------------------------------------------------------------------
# Copyright (c) 2016-2020, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import csv

import qiime2.plugin.model as model
from qiime2.plugin import ValidationError
from plugin_setup import plugin


class ResultsFormat(model.TextFileFormat):
    def _validate_(self, level):
        with self.open() as fh:
            try:
                csv.reader(fh, delimiter='\t')
            except ValidationError:
                raise ValidationError()


ResultsDirectoryFormat = model.SingleFileDirectoryFormat(
    'ResultsDirectoryFormat', 'results.tsv',
    ResultsFormat)

plugin.register_formats(ResultsFormat, ResultsDirectoryFormat)
