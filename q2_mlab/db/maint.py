import json
import pandas as pd
from sqlalchemy.engine import Engine
from sqlalchemy import create_engine as sql_create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from qiime2 import Artifact
from q2_mlab.db.schema import RegressionScore, Parameters, Base
from q2_mlab.db.mapping import remap_parameters
from typing import Optional, Callable


DROP_COLS = ['SAMPLE_ID', 'Y_PRED', 'Y_TRUE']


def format_db(db_file: Optional[str] = None) -> str:
    if db_file is not None:
        loc = '/' + db_file
    else:
        # this creates an in memory database
        loc = ''
    return f"sqlite://{loc}"


def create_engine(db_file: Optional[str] = None, echo=True) -> Engine:
    engine = sql_create_engine(format_db(db_file), echo=echo)
    return engine


def create(db_file: Optional[str] = None, echo=True) -> Engine:
    engine = create_engine(db_file, echo=echo)
    Base.metadata.create_all(engine)
    return engine


def add(engine: Engine, results: pd.DataFrame, parameters: dict,
        dataset: str, target: str, level: str, algorithm: str,
        ) -> None:
    Session = sessionmaker(bind=engine)
    session = Session()

    params = Parameters(**parameters)
    session.add(params)
    session.flush()

    for entry in results.iterrows():
        score = RegressionScore(datetime=datetime.now(),
                                parameters_id=params.id,
                                dataset=dataset,
                                target=target,
                                level=level,
                                algorithm=algorithm,
                                **entry[1],
                                )
        session.add(score)

    session.commit()


def add_from_qza(artifact_path: str,
                 parameters_string: str,
                 dataset: str,
                 target: str,
                 level: str,
                 algorithm: str,
                 db_file: Optional[str] = None,
                 echo: bool = True,
                 engine_creator: Callable = create_engine,
                 ):

    engine = engine_creator(db_file, echo=echo)

    results = Artifact.load(artifact_path).view(pd.DataFrame)
    # perform filtering on results (single entry per cross validation)
    results.drop(DROP_COLS, axis=1, inplace=True)
    results.drop_duplicates(inplace=True)

    parameters = json.loads(parameters_string)
    # remap multi-type arguments, e.g., 'max_features'
    parameters = remap_parameters(parameters)

    add(engine, results, parameters,
        dataset=dataset, target=target, level=level, algorithm=algorithm,
        )
    return engine
