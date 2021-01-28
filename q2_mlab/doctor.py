#!/usr/bin/env python
import click
import os
from datetime import datetime
import pandas as pd


def sort_result_artifact_filenames(list_of_artifact_filenames):
    """Sort the result artifact filenames

    Sorts the given list of result filenames by parameter index (assumed to be
    the beginning of the filename, preceding an underscore e.g.
    ``00004_*.qza``)

    Parameters
    ----------
    list_of_artifact_filenames : List
        A list of artifact filenames returned by the ``get_results`` method

    Returns
    -------
    sorted_artifact_filenames : List
        Sorted list of the found artifact filenames

    See Also
    --------
    q2_mlab.doctor.get_results

    """
    sorted_artifact_filenames = sorted(
        list_of_artifact_filenames, key=lambda name: int(name.split("_")[0])
    )
    return sorted_artifact_filenames


def parse_info(info_filepath):
    return pd.read_csv(info_filepath, skipinitialspace=True).to_dict(
        "records"
    )[0]


def get_results(results_dir):
    """Get result artifact filenames

    Scans the given directory and returns a list of filenames of all QIIME2
    artifacts in the directory.

    Parameters
    ----------
    results_dir : str
        The directory containing result artifacts.

    Returns
    -------
    artifact_filenames : List
        A List of the found artifact filenames.

    """
    if os.path.isdir(results_dir):
        artifact_filenames = [
            f.name for f in os.scandir(results_dir) if f.path.endswith(".qza")
        ]
    else:
        artifact_filenames = []
    return artifact_filenames


def filter_duplicate_parameter_results(
    list_of_artifact_filenames, results_dir, delete=False
):
    """Filter out duplicate results on the same parameter set

    From a sorted list of artifact filenames (sorted by parameter index),
    compare neighboring artifacts and if they are duplicate results on the same
    parameter index, either ignore (delete=False) or delete (delete=True) the
    older of the two duplicate results.

    Parameters
    ----------
    list_of_artifact_filenames : List
        A list of artifact filenames returned by get_results()
    results_dir : str
        The directory containing result artifacts.
    delete : bool, optional
        Delete the duplicate results if ``True``, or ignore them if
        ``False``.

    Returns
    -------
    artifact_filenames : List
        List of the artifact filenames, excluding duplicated results.

    """

    artifact_filenames = list_of_artifact_filenames.copy()

    # Change working dirs
    prev_working_dir = os.getcwd()
    os.chdir(results_dir)

    if len(artifact_filenames) == 0:
        raise ValueError("There are no result artifacts to remove.")
    artifact_filenames = sort_result_artifact_filenames(artifact_filenames)

    curr_path = artifact_filenames[0]
    curr_param_idx = int(curr_path.split("_")[0])

    # From the list of filenames (sorted by parameter index), check if neighboring
    # filenames are of the same parameter index.
    for next_path in artifact_filenames[1:]:
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
                artifact_filenames.remove(curr_path)
                if delete:
                    os.remove(curr_path)
            else:
                artifact_filenames.remove(next_path)
                if delete:
                    os.remove(next_path)
                continue  # Do not update current_path
        # Update current path
        curr_path = next_path
        curr_param_idx = next_param_idx

    # Change working dir back
    os.chdir(prev_working_dir)

    return artifact_filenames


def doctor_hyperparameter_search(
    dataset,
    preparation,
    target,
    algorithm,
    base_dir,
    max_results=1000,
    delete_duplicates=False,
):
    """Identify jobs to relaunch to generate missing result artifacts

    Searches for the job description file for the experiment described by
    this function's parameters, searches through the files produced for
    missing results, and relaunches jobs necessary for completing those results.

    Parameters
    ----------
    dataset : str
        Name of dataset.
    preparation : str
        Name of data type/preparation (e.g. 16S).
    target : str
        Name of the target variable in the metadata.
    algorithm : str
        Valid algorithm included in q2_mlab.
    base_dir : str
        The directory in which create the file structure.
    max_results : str, optional
        The maximum number of result artifacts to expect. Defaults to 1000.
    delete_duplicates : bool, optional
        If ``True``, deletes the older of duplicate results on the same
        parameter id. Defaults to ``False`` where duplicate results are
        ignored.

    Returns
    -------
    cmd : str
        Command to regenerate missing artifacts.

    Raises
    ------
    FileNotFoundError
        If ``base_dir`` does not exist.
    FileNotFoundError
        If directory with results for this experiement does not exist.
    FileNotFoundError
        If the job script for this experiment does not exist.
    FileNotFoundError
        If the info file for this experiment does not exist.

    See Also
    --------
    q2_mlab.orchestrator.orchestrate_hyperparameter_search

    """
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
        msg = "Cannot find directory with result file structure. This is typically generated with 'mlab-orchestrator'.\n"
        raise FileNotFoundError(msg + base_dir + " does not exist.")

    if not os.path.isdir(results_dir):
        msg = "Cannot find directory with results. This is typically generated with 'mlab-orchestrator' or 'qiime mlab unit_benchmark'\n"
        raise FileNotFoundError(msg + results_dir + " does not exist.")

    if not os.path.exists(output_script):
        msg = "Cannot find the job script for this experiment. This is typically generated with 'mlab-orchestrator'\n"
        raise FileNotFoundError(msg + output_script + " does not exist.")

    if not os.path.exists(info_doc):
        msg = "Cannot find info file for this experiment. This is typically generated with 'mlab-orchestrator'\n"
        raise FileNotFoundError(msg + info_doc + " does not exist.")

    # Parse info doc for expected result info
    info_dict = parse_info(info_doc)

    # Get all filenames in the results directory
    uninserted_filenames = get_results(results_dir)
    inserted_dir = os.path.join(results_dir, "inserted")
    inserted_filenames = get_results(inserted_dir)

    # Remove duplicate parameter indices
    if len(inserted_filenames) > 0:
        inserted_filenames = filter_duplicate_parameter_results(
            inserted_filenames, inserted_dir, delete=delete_duplicates
        )
    if len(uninserted_filenames) > 0:
        uninserted_filenames = filter_duplicate_parameter_results(
            uninserted_filenames, results_dir, delete=delete_duplicates
        )

    # Compute missing parameter indices:
    expected_num_results = min(max_results, info_dict["parameter_space_size"])
    expected_param_indices = set(range(1, expected_num_results + 1))

    all_filenames = inserted_filenames + uninserted_filenames
    all_param_indices = {int(f.split("_")[0]) for f in all_filenames}

    missing_param_indices = expected_param_indices - all_param_indices

    if len(missing_param_indices) == 0:
        return None

    chunks_to_rerun = set()
    for missing_idx in missing_param_indices:
        # Identify which chunk to run:
        chunk_size = info_dict["chunk_size"]
        missing_chunk = ((missing_idx - 1) // chunk_size) + 1
        chunks_to_rerun.add(missing_chunk)

    # Return command to re-run missing chunks
    cmd = f"qsub -t {','.join(map(str, chunks_to_rerun))} {output_script}"
    print(cmd)
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
