# ----------------------------------------------------------------------------
# Copyright (c) 2016-2020, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#qiime dev refresh-cache
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from ._preprocess import preprocess
from ._benchmark import benchmark_classify
from ._version import get_versions

__version__ = get_versions()['version']
del get_versions

__all__ = ['preprocess', 'benchmark_classify']
