import unittest
import os
from q2_mlab import _orchestrate


class OrchestratorTests(unittest.TestCase):

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

        target = "reported-antibiotic-usage"
        dataset = "imsms_test"
        prep = "16S"
        alg = "LinearSVC"

        script_fp, params_fp, run_info_fp = _orchestrate(
            dataset=dataset,
            preparation=prep,
            target=target,
            algorithm=alg,
            base_dir=self.TEST_DIR,
            dataset_path=self.dataset_file,
            metadata_path=self.metadata_file,
            dry=True
        )

        print(script_fp, params_fp, run_info_fp)
        self.assertEqual(
            script_fp, 
            os.path.join(
                self.TEST_DIR,
                f"{dataset}/{prep}_{target}_{alg}.sh"
            )
        )
        expected = f"{dataset}/{prep}/{target}/{alg}/{alg}_parameters.txt"
        self.assertEqual(
            params_fp, 
            os.path.join(
                self.TEST_DIR,
                expected
            )
        )
        expected = f"{dataset}/{prep}_{target}_{alg}_info.txt"
        self.assertEqual(
            run_info_fp, 
            os.path.join(
                self.TEST_DIR,
                expected
            )
        )
