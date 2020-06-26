# ----------------------------------------------------------------------------
# Copyright (c) 2020, mlab development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import csv

import qiime2.plugin.model as model
from qiime2.plugin import ValidationError


class ResultsFormat(model.TextFileFormat):

    def _validate_(self, level):
        with self.open() as fh:
            try:
                csv.reader(fh, delimiter='\t')
            except ValidationError:
                raise ValidationError()


ResultsDirectoryFormat = model.SingleFileDirectoryFormat(
    'ResultsDirectoryFormat', 'results.tsv', ResultsFormat)
