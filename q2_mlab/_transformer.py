from . import ResultsFormat
import pandas as pd
from .plugin_setup import plugin


@plugin.register_transformer
def _2(data: pd.DataFrame) -> ResultsFormat:
    ff = ResultsFormat()
    with ff.open() as fh:
        data.to_csv(fh, sep='\t', header=True, index=False)
    return ff


@plugin.register_transformer
def _3(ff: ResultsFormat) -> pd.DataFrame:
    with ff.open() as fh:
        df = pd.read_csv(fh, sep="\t")
    return df
