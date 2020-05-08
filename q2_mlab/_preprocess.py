import numpy as np
import pandas as pd
import biom
import skbio
from biom import load_table
from qiime2 import Artifact

def print_datasize(table, metadata):
    biom_table = table.view(biom.Table)
    dataframe = metadata.to_dataframe()
    print("\nTable Shape: " + str(biom_table.shape))
    print("Metadata Shape: " + str(dataframe.shape) + "\n")

def preprocess(ctx, table, metadata, phylogeny, sampling_depth,
                           min_frequency, target_variable, with_replacement,
                           n_jobs=1):
    
    # Define Qiime methods to call
    rarefy = ctx.get_action('feature_table', 'rarefy')
    filter_features = ctx.get_action('feature_table', 'filter_features')
    filter_samples = ctx.get_action('feature_table', 'filter_samples')
    beta = ctx.get_action('diversity', 'beta')
    beta_phylogenetic = ctx.get_action('diversity', 'beta_phylogenetic')
    filter_distance_matrix = ctx.get_action('diversity', 'filter_distance_matrix')

    #filter_ids = ctx.get_action('metadata', 'filter_ids')

    results = []
    print_datasize(table, metadata)

    # Filter low-abundance features from table
    filtered_table, = filter_features(table=table, min_frequency=min_frequency)
    print_datasize(filtered_table, metadata)

    # Rarefy table to sampling_depth
    rarefied_table, = rarefy(table=filtered_table, sampling_depth=sampling_depth,
                             with_replacement=with_replacement)
    print_datasize(rarefied_table, metadata)

    # Filter samples from metadata where NaN in target_variable column 
    # TODO

    # Filter metadata by samples in table
    ids_to_keep = rarefied_table.view(biom.Table).ids()
    filtered_metadata = metadata.filter_ids(ids_to_keep=ids_to_keep)
    print_datasize(rarefied_table, filtered_metadata)

    # Filter table by samples in metadata 
    filtered_rarefied_table_results = filter_samples(table=rarefied_table, 
                                                     metadata=filtered_metadata)
    filtered_rarefied_table = filtered_rarefied_table_results.filtered_table
    print_datasize(filtered_rarefied_table, filtered_metadata)

    results += filtered_rarefied_table_results

    # TODO append metadata to results, somehow i.e.
    # results += metadata_visualization

    # Generate Distance Matrices
    for metric in ['jaccard', 'braycurtis', 'jensenshannon', 'aitchison'] :
        beta_results = beta(table=filtered_rarefied_table, metric=metric, 
                            n_jobs=n_jobs)
        results += beta_results
    for metric in ['unweighted_unifrac', 'weighted_unifrac']:
        beta_phylo_results = beta_phylogenetic(table=filtered_rarefied_table, 
                                               phylogeny=phylogeny,
                                               metric=metric)
        results += beta_phylo_results

    # TODO Filter table based on ids preserved in phylo distance matrices

    return tuple(results)