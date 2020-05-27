import pandas as pd
import biom
from qiime2 import Metadata


def print_datasize(table, metadata):
    biom_table = table.view(biom.Table)
    dataframe = metadata.to_dataframe()
    print("\nTable Shape: " + str(biom_table.shape))
    print("Metadata Shape: " + str(dataframe.shape) + "\n")


def clean_metadata(df: pd.DataFrame, target_variable, discrete):
    # metadata categories that are used in this pipeline are assumed to be
    # cleaned upfront. For classification, the only allowed values are 0 and 1.
    # For regression, the allowed values are any real numbers
    # Enforce numeric for continuous variables
    df[target_variable] = pd.to_numeric(df[target_variable],
                                        errors='coerce')

    subset_df = df.loc[:, [target_variable]]
    subset_df.dropna(axis=0, subset=[target_variable], inplace=True)

    if discrete:
        val_set = set(subset_df[target_variable].unique())
        if not val_set == {1, 0}:
            raise ValueError('For classification, the only allowed values '
                             'in the target column are 0 and 1')
    return subset_df


def preprocess(ctx, table, metadata, sampling_depth, min_frequency,
               target_variable, discrete, with_replacement=False, n_jobs=1):

    # Define QIIME2 methods to call
    rarefy = ctx.get_action('feature_table', 'rarefy')
    filter_min_features = ctx.get_action('feature_table', 'filter_features')
    filter_samples = ctx.get_action('feature_table', 'filter_samples')
    beta = ctx.get_action('diversity', 'beta')
    results = []
    print("Initial datasize:")
    print_datasize(table, metadata)

    print("Filtering table by samples in metadata")
    # Filter table by samples present in metadata
    filtered_table, = filter_samples(table=table, metadata=metadata)
    # Filter metadata by samples in table
    ids_to_keep = filtered_table.view(biom.Table).ids()
    filteredmetadata = metadata.filter_ids(ids_to_keep=ids_to_keep)

    # Filter samples from metadata where NaN in target_variable column
    # Reduce Metadata to 1 column mapping of sample-id to target
    df = filteredmetadata.to_dataframe()
    clean_subset_df = clean_metadata(df=df, target_variable=target_variable,
                                     discrete=discrete)
    target_mapping = Metadata(clean_subset_df)

    print("Filtering low-abundance features from table:")
    # Filter low-abundance features from table
    filtered_table, = filter_min_features(table=filtered_table,
                                          min_frequency=min_frequency)
    print_datasize(filtered_table, metadata)

    print("Rarefying table to sampling_depth")
    # Rarefy table to sampling_depth
    rarefied_table, = rarefy(table=filtered_table,
                             sampling_depth=sampling_depth,
                             with_replacement=with_replacement)
    print_datasize(rarefied_table, metadata)

    print("Filtering table by samples in metadata")
    # Filter table by samples in metadata
    filtered_rarefied_table_results = filter_samples(table=rarefied_table,
                                                     metadata=filteredmetadata)
    filtered_rarefied_table = filtered_rarefied_table_results.filtered_table
    print_datasize(filtered_rarefied_table, filteredmetadata)

    results += filtered_rarefied_table_results

    print("Refiltering target_mapping by samples in table")
    # Refilter target_mapping by samples in table
    ids_to_keep = filtered_rarefied_table.view(biom.Table).ids()
    target_mapping = target_mapping.filter_ids(ids_to_keep=ids_to_keep)

    print("Some transformations to get data into correct format for artifact")
    # Some transformations to get data into correct format for artifact
    target_mapping_col = target_mapping.get_column(target_variable)
    target_mapping_series = target_mapping_col.to_series()
    target_mapping_artifact = ctx.make_artifact("SampleData[Target]",
                                                target_mapping_series)
    results += [target_mapping_artifact]

    print("Not generating any distance matrices...")
    
    return tuple(results)
