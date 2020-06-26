# ----------------------------------------------------------------------------
# Copyright (c) 2020, mlab development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import unittest

from q2_mlab.plugin_setup import plugin as mlab_plugin


class PluginSetupTests(unittest.TestCase):

    def test_plugin_setup(self):
        self.assertEqual(mlab_plugin.name, 'mlab')
