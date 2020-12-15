import unittest
import os
from q2_mlab import _orchestrate


class OrchestratorTests((unittest.TestCase)):

    def setUp(self):
        self.TEST_DIR = os.path.split(__file__)[0]
        self.dataset_file = os.path.join(
            self.TEST_DIR, "data/table.qza"
        )
        self.metadata_file = os.path.join(
            self.TEST_DIR, "data/sample-metadata-binary.tsv"
        )

    def testDryRun(self):
        print(__file__)
        print(os.getcwd())
        _orchestrate(
            dataset="imsms_test",
            preparation="16S",
            target="reported-antibiotic-usage",
            algorithm="LinearSVC",
            base_dir=self.TEST_DIR,
            dataset_path=self.dataset_file,
            metadata_path=self.metadata_file,
            dry=True
        )
