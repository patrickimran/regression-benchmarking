import unittest
import os
import shutil
import subprocess
import pandas as pd
from q2_mlab import _orchestrate
from pandas.testing import assert_frame_equal



class OrchestratorTests(unittest.TestCase):

    def setUp(self):
        self.dataset = "dset_test"
        self.TEST_DIR = os.path.split(__file__)[0]
        self.dataset_file = os.path.join(
            self.TEST_DIR, "data/table.qza"
        )

        # The metadata doesn't matter here, as mlab will only accept 
        # SampleData[Target] artifacts from preprocessing.
        self.metadata_file = os.path.join(
            self.TEST_DIR, "data/sample-metadata.tsv"
        )

        target = "reported-antibiotic-usage"
        dataset = self.dataset
        prep = "16S"
        alg = "LinearSVR"

        self.script_fp, self.params_fp, self.run_info_fp = _orchestrate(
            dataset=dataset,
            preparation=prep,
            target=target,
            algorithm=alg,
            base_dir=self.TEST_DIR,
            dataset_path=self.dataset_file,
            metadata_path=self.metadata_file,
            chunk_size=20,
            dry=False
        )

        # Make a runnable test script, removing TORQUE job info
        self.test_script = os.path.splitext(self.script_fp)[0] + "_test.sh"
        with open(self.script_fp) as f:
            # Keep last 49 lines
            keeplines = f.readlines()[-49:]
            with open(self.test_script, "w") as out:
                for line in keeplines:
                    out.write(line)
        subprocess.run(["chmod", "755", self.test_script])
    
    def tearDown(self):
        if (self.script_fp and os.path.exists(self.script_fp)):
            os.remove(self.script_fp)
        if (self.params_fp and os.path.exists(self.params_fp)):
            os.remove(self.params_fp)
        if (self.run_info_fp and os.path.exists(self.run_info_fp)):
            os.remove(self.run_info_fp)
        
        # Remove the directory structure created by Orchestrator
        if os.path.exists(os.path.join(self.TEST_DIR, self.dataset)):
            shutil.rmtree(os.path.join(self.TEST_DIR, self.dataset))

    def testDryRun(self):

        target = "reported-antibiotic-usage"
        dataset = self.dataset
        prep = "16S"
        alg = "LinearSVR"

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

    def testRun(self):

        self.assertTrue(os.path.exists(self.script_fp))
        self.assertTrue(os.path.exists(self.params_fp))
        self.assertTrue(os.path.exists(self.run_info_fp))

        expected_run_info = [{
            "parameters_fp": self.params_fp,
            "parameter_space_size": 112,
            "chunk_size": 20, 
            "remainder": 12,
            "n_chunks": 6,
        }]
        expected_run_info_df = pd.DataFrame.from_records(expected_run_info)
        returned_run_info_df = pd.read_csv(self.run_info_fp)
        assert_frame_equal(
            expected_run_info_df,
            returned_run_info_df,
            check_dtype=False
        )
        returned_num_params = -1
        with open(self.params_fp) as f:
            returned_num_params = sum(1 for line in f)
        self.assertEqual(112, returned_num_params)

    def testInvalidJobArrayID(self):
        
        # Test invalid array ids
        os.environ["PBS_ARRAYID"] = "7"
        completed_process = subprocess.run(
            f"{self.test_script}",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        self.assertEqual(completed_process.returncode, 0)
        print(completed_process)

        os.environ["PBS_ARRAYID"] = "50"
        completed_process = subprocess.run(
            f"{self.test_script}",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        self.assertEqual(completed_process.returncode, 0)
        print(completed_process)

    def testValidJobArrayID(self):
        # Since the metadata provided is not SampleData[Target], 
        # we expect q2_mlab to throw an error with return code 1.
        # The downside to this over checking for a timeout is that waiting
        # for 20 (n_chunks) errors takes time.

        os.environ["PBS_ARRAYID"] = "5"
        completed_process = subprocess.run(
            f"{self.test_script}",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(completed_process.returncode, 1)
        print(completed_process)

        os.environ["PBS_ARRAYID"] = "6"
        completed_process = subprocess.run(
            f"{self.test_script}",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(completed_process.returncode, 1)
        print(completed_process)
