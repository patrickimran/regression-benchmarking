# ----------------------------------------------------------------------------
# Copyright (c) 2016-2020, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from qiime2.plugin import SemanticType
from q2_types.sample_data import SampleData
from plugin_setup import plugin
from . import ResultsDirectoryFormat

Results = SemanticType('Results', variant_of=SampleData.field['type'])

plugin.register_semantic_types(Results)

plugin.register_semantic_type_to_format(
    SampleData[Results],
    artifact_format=ResultsDirectoryFormat
)
