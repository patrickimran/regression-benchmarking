#!/usr/bin/env python
import click
import os
from q2_mlab import (
     parse_info,
     get_results,
     filter_duplicate_parameter_results,
 )
from q2_mlab.db.maint import (
    create as db_create,
    add_from_qza, 
    add
)
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from q2_mlab.db.schema import RegressionScore, ClassificationScore, Parameters, Base

def collect_hyperparameter_search(
     dataset,
     preparation,
     target,
     algorithm,
     base_dir,
     database_file,
     create_db=False
 ):
    """
    Trawls through the result files for the configuration described by
    this function's parameters, and inserts results into the given SQLite3
    database.
    
            Parameters:
                    dataset (str): Name of dataset
                    preparation (str): Name of data type/preparation (e.g. 16S)
                    target (str): Name of the target variable in the metadata
                    algorithm (str): Valid algorithm included in q2_mlab
                    base_dir (str): The directory in which create the file structure.
                    database_file (str): Path to the database file.
                    create_db (bool): If true, create a new database at the given filepath. Default: False
            Returns:
                    None
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
    "--database",
    "-db",
    help="Path to the existing database file",
    required=True,
)
@click.option(
    "--create-db/--no-create-db",
    default=False,
    show_default=True,
    help="Create a new database at the given filepath if it does not exist",
)
def cli(
    dataset,
    preparation,
    target,
    algorithm,
    base_dir,
    database,
    create_db,
):
    collect_hyperparameter_search(
    dataset=dataset,
    preparation=preparation,
    target=target,
    algorithm=algorithm,
    base_dir=base_dir,
    database_file=database,
    create_db=create_db
)