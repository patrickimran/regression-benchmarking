#!/usr/bin/env python
import click
import os
from datetime import datetime


def parse_info(info_filepath):
    with open(info_filepath) as f:
        lines = f.readlines()
        info_dict = {
            key.strip(): val.strip()
            for key, val in zip(lines[0].split(","), lines[1].split(","))
        }
    return info_dict


def get_uninserted_results(results_dir):
    return [f.name for f in os.scandir(results_dir) if f.path.endswith(".qza")]


def get_inserted_results(results_dir):
    inserted_dir = os.path.join(results_dir, "inserted")
    if not os.path.isdir(inserted_dir):
        return []
    else:
        return [
            f.name for f in os.scandir(inserted_dir) if f.path.endswith(".qza")
        ]


def filter_duplicate_parameter_results(
    list_of_artifact_paths, results_dir, delete=False
):
    artifact_paths = list_of_artifact_paths.copy()
    os.chdir(results_dir)
    if len(artifact_paths) == 0:
        raise ValueError("There are no result artifacts to remove.")
    artifact_paths.sort()

    curr_path = artifact_paths[0]
    curr_param_idx = int(curr_path.split("_")[0])

    for next_path in artifact_paths[1:]:
        next_param_idx = int(next_path.split("_")[0])
        if next_param_idx == curr_param_idx:
            # Remove the older of the duplicate results
            curr_path_time = datetime.fromtimestamp(
                os.stat(curr_path).st_mtime
            )
            next_path_time = datetime.fromtimestamp(
                os.stat(next_path).st_mtime
            )
            if curr_path_time < next_path_time:
                artifact_paths.remove(curr_path)
                if delete:
                    os.remove(curr_path)
            else:
                artifact_paths.remove(next_path)
                if delete:
                    os.remove(next_path)
                continue  # Do not update current_path
        # Update current path
        curr_path = next_path
        curr_param_idx = next_param_idx

    return artifact_paths


def doctor_hyperparameter_search(
    dataset,
    preparation,
    target,
    algorithm,
    base_dir,
    max_results=1000,
    delete_duplicates=False,
):
    """
    Searches for the job description file for the experiment described by
    this function's parameters, searches through the files produced for
    missing results, and relaunches jobs necessary for completing those results.

            Parameters:
                    dataset (str): Name of dataset
                    preparation (str): Name of data type/preparation (e.g. 16S)
                    target (str): Name of the target variable in the metadata
                    algorithm (str): Valid algorithm included in q2_mlab
                    base_dir (str): The directory in which create the file structure.
                    max_results (str): The maximum number of result artifacts to expect.
                    delete_duplicates (bool): If True, deletes the older of duplicated results on the same parameter id.
            Returns:
                    summary (str): Summary of errors and jobs relaunched/to be relaunched
    """
    barnacle_out_dir = os.path.join(base_dir, dataset, "barnacle_output/")
    results_dir = os.path.join(
        base_dir, dataset, preparation, target, algorithm
    )
    output_script = os.path.join(
        base_dir, dataset, "_".join([preparation, target, algorithm]) + ".sh"
    )
    info_doc = os.path.join(
        base_dir,
        dataset,
        "_".join([preparation, target, algorithm]) + "_info.txt",
    )

    if not os.path.isdir(base_dir):
        msg = "Cannot find directory with result file structure. This is typically generated with 'orchestrator'.\n"
        raise FileNotFoundError(msg + base_dir + " does not exist.")

    if not os.path.isdir(barnacle_out_dir):
        msg = "Cannot find directory with job output. This is typically generated with 'orchestrator'.\n"
        raise FileNotFoundError(msg + barnacle_out_dir + " does not exist.")

    if not os.path.isdir(results_dir):
        msg = "Cannot find directory with results. This is typically generated with 'qiime mlab unit_benchmark'\n"
        raise FileNotFoundError(msg + results_dir + " does not exist.")

    if not os.path.exists(output_script):
        msg = "Cannot find the job script for this experiment. This is typically generated with 'orchestrator'\n"
        raise FileNotFoundError(msg + output_script + " does not exist.")

    if not os.path.exists(info_doc):
        msg = "Cannot find info file for this experiment. This is typically generated with 'orchestrator'\n"
        raise FileNotFoundError(msg + info_doc + " does not exist.")

    # Parse info doc for expected result info
    info_dict = parse_info(info_doc)

    # Get all filenames in the results directory
    uninserted_filenames = get_uninserted_results(results_dir)
    inserted_filenames = get_inserted_results(results_dir)

    # TODO remove duplicate parameter indices
    if len(inserted_filenames) > 0:
        inserted_dir = os.path.join(results_dir, "inserted")
        inserted_filenames = filter_duplicate_parameter_results(
            inserted_filenames, inserted_dir, delete=delete_duplicates
        )
    if len(uninserted_filenames) > 0:
        uninserted_filenames = filter_duplicate_parameter_results(
            uninserted_filenames, results_dir, delete=delete_duplicates
        )

    # Compute missing parameter indices:
    expected_num_results = min(
        max_results, int(info_dict["parameter_space_size"])
    )
    expected_param_indices = set(range(1, expected_num_results + 1))

    all_filenames = inserted_filenames + uninserted_filenames
    all_param_indices = {int(f.split("_")[0]) for f in all_filenames}

    missing_param_indices = expected_param_indices - all_param_indices

    if missing_param_indices == 0:
        return None

    chunks_to_rerun = set()
    for missing_idx in missing_param_indices:
        # Identify which chunk to run:
        chunk_size = int(info_dict["chunk_size"])
        missing_chunk = (missing_idx // chunk_size) + 1
        chunks_to_rerun.add(missing_chunk)

    # Return command to re-run missing chunks
    cmd = f"qsub -t {','.join(map(str, chunks_to_rerun))} {output_script}"
    print(cmd)
    # subprocess.run(cmd)
    return cmd


@click.command()
@click.argument("dataset")
@click.argument("preparation")
@click.argument("target")
@click.argument("algorithm")
@click.option(
    "--base_dir",
    "-b",
    help="Directory to search for datasets in",
    required=True,
)
@click.option(
    "--max_results",
    default=1000,
    show_default=True,
    help="Max number of result artifacts we expect per algorithm.",
)
@click.option(
    "--delete-duplicates/--no-delete-duplicates",
    default=False,
    show_default=True,
    help="Remove duplicate result artifacts.",
)
def cli(
    dataset,
    preparation,
    target,
    algorithm,
    base_dir,
    max_results,
    delete_duplicates,
):
    doctor_hyperparameter_search(
        dataset=dataset,
        preparation=preparation,
        target=target,
        algorithm=algorithm,
        base_dir=base_dir,
        max_results=max_results,
        delete_duplicates=delete_duplicates,
    )
