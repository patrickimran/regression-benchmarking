from qiime2.plugin import (Plugin, Str, Int, Bool, Range, Visualization,
                           Metadata, Citations, SemanticType, Choices)
import pandas as pd
import importlib
import q2_mlab
from q2_types.feature_table import FeatureTable, Frequency
from q2_types.distance_matrix import DistanceMatrix
from q2_types.sample_data import SampleData
from q2_types.tree import Phylogeny, Rooted
from q2_sample_classifier import PredictionsDirectoryFormat
from q2_mlab._type import Target, Results
from q2_mlab._format import ResultsFormat, ResultsDirectoryFormat

citations = Citations.load('citations.bib', package='q2_mlab')

sklearn_n_jobs_description = (
    'The number of jobs to use for the computation. This works by breaking '
    'down the pairwise matrix into n_jobs even slices and computing them in '
    'parallel. If -1 all CPUs are used. If 1 is given, no parallel computing '
    'code is used at all, which is useful for debugging. For n_jobs below -1, '
    '(n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one '
    'are used. (Description from sklearn.metrics.pairwise_distances)'
)

plugin = Plugin(
    name='mlab',
    version=q2_mlab.__version__,
    website='https://dev.qiime2.org/',
    package='q2_mlab',
    citations=Citations.load('citations.bib', package='q2_mlab'),
    description=('This QIIME 2 plugin is in development'
                 ' and does sweet stuff.'),
    short_description='Plugin for machine learning automated benchmarking.'
)

plugin.register_formats(ResultsFormat, ResultsDirectoryFormat)

plugin.register_semantic_types(Target)
plugin.register_semantic_type_to_format(
    SampleData[Target],
    artifact_format=PredictionsDirectoryFormat
)

plugin.register_semantic_types(Results)
plugin.register_semantic_type_to_format(
    SampleData[Results],
    artifact_format=ResultsDirectoryFormat
)

plugin.pipelines.register_function(
    function=q2_mlab.preprocess,
    inputs={
        'table': FeatureTable[Frequency],
        'phylogeny': Phylogeny[Rooted]
    },
    parameters={
        'metadata': Metadata,
        'sampling_depth': Int % Range(1, None),
        'min_frequency': Int % Range(1, None),
        'target_variable': Str,
        'discrete': Bool,
        'with_replacement': Bool,
        'n_jobs': Int % Range(-1, None),
    },
    outputs=[
        ('filtered_rarefied_table', FeatureTable[Frequency]),
        ('filtered_metadata', SampleData[Target]),
        ('jaccard_distance_matrix', DistanceMatrix),
        ('bray_curtis_distance_matrix', DistanceMatrix),
        ('jensenshannon_distance_matrix', DistanceMatrix),
        ('aitchison_distance_matrix', DistanceMatrix),
        ('unweighted_unifrac_distance_matrix', DistanceMatrix),
        ('weighted_unifrac_distance_matrix', DistanceMatrix),
    ],
    input_descriptions={
        'table': 'The 16S or metagenomic feature table.',
        'phylogeny': 'Phylogenetic tree containing tip identifiers that '
                     'correspond to the feature identifiers in the table. '
                     'This tree can contain tip ids that are not present in '
                     'the table, but all feature ids in the table must be '
                     'present in this tree.'
    },
    parameter_descriptions={
        'metadata': 'The sample metadata (tsv) used to filter the table and '
                    'containing a column for the target variable.',
        'sampling_depth': 'The total frequency that each sample should be '
                          'rarefied to prior to computing diversity metrics.',
        'min_frequency': 'The minimum frequency that a feature should have '
                         'to be maintained.',
        'target_variable': 'The metadata column containing the variable of '
                           'interest for this learning task.',
        'discrete': 'Set True if target_variable is a discrete variable '
                    '(for Classification), False if continuous (Regression)',
        'with_replacement': 'Rarefy with replacement by sampling from the '
                            'multinomial distribution instead of rarefying '
                            'without replacement.',
        'n_jobs': sklearn_n_jobs_description
    },
    output_descriptions={
        'filtered_rarefied_table': 'The resulting filtered and rarefied '
                                   'feature table.',
        'filtered_metadata': 'The resulting filtered metadata containing '
                             'only the specific target column.',
        'jaccard_distance_matrix':
            'Matrix of Jaccard distances between pairs of samples.',
        'bray_curtis_distance_matrix':
            'Matrix of Bray-Curtis distances between pairs of samples.',
        'jensenshannon_distance_matrix':
            'Matrix of Jensen-Shannon distances between pairs of samples.',
        'aitchison_distance_matrix':
            'Matrix of aitchison distances between pairs of samples.',
        'unweighted_unifrac_distance_matrix':
            'Matrix of unweighted UniFrac distances between pairs of samples.',
        'weighted_unifrac_distance_matrix':
            'Matrix of weighted UniFrac distances between pairs of samples.',
    },
    name='Dataset preprocessing for benchmarking',
    description=('Applies filtering and preprocessing steps '
                 'to a feature table and metadata, and '
                 'generates distance matrices with phylo- '
                 'genetic and non-phylogenetic metrics.')
)

algorithm_choices = [
    "KNeighborsClassifier",
    "RandomForestClassifier",
    "GradientBoostingClassifier",
    "XGBClassifier",
    "LinearSVC",

    "KNeighborsRegressor",
    "RandomForestRegressor",
    "GradientBoostingRegressor",
    "XGBRegressor",
    "LinearSVR",
]

plugin.methods.register_function(
    function=q2_mlab.unit_benchmark,
    inputs={
        'table': FeatureTable[Frequency],
        'distance_matrix': DistanceMatrix,
        'metadata': SampleData[Target]
    },
    parameters={
        'algorithm': Str % Choices(algorithm_choices),
        'params': Str,
        'n_jobs': Int % Range(-1, None),
    },
    outputs=[
        ('result_table', SampleData[Results]),
    ],
    input_descriptions={
        'table': 'The feature table containing the samples over which '
                 'diversity metrics should be computed.',
        'distance_matrix': 'Matrix of pairwise distances between samples in '
                           'the given table.',
        'metadata': 'The sample metadata used filter the table and containing '
                    'a column for the target variable.'
    },
    parameter_descriptions={
        'algorithm': 'Name of the Scikit-Learn supervised learning model to '
                     'use. Classification or Regression will be determined '
                     'according to the given algorithm.',
        'params': 'The input parameters as a JSON string.',
        'n_jobs': sklearn_n_jobs_description
    },
    output_descriptions={
        'result_table': 'Output predictions and performance measures.',
    },
    name='Unit benchmarking script for evaluating one model/hyperparameter set',
    description=('Conducts cross validation and computation of metrics for '
                 'evaluating one model with one hyperparameter set, given as '
                 'input. ')
)

importlib.import_module('q2_mlab._transformer')