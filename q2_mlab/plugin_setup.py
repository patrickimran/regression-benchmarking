from qiime2.plugin import (Plugin, Str, Int, Bool, Range, Visualization,
                           Metadata, Citations, SemanticType)
import q2_mlab
from q2_sample_classifier import PredictionsDirectoryFormat
from q2_types.feature_table import FeatureTable, Frequency
from q2_types.distance_matrix import DistanceMatrix
from q2_types.sample_data import SampleData
from q2_types.tree import Phylogeny, Rooted

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

Target = SemanticType('Target', variant_of=SampleData.field['type'])
plugin.register_semantic_types(Target)
plugin.register_semantic_type_to_format(
    SampleData[Target],
    artifact_format=PredictionsDirectoryFormat
)


plugin.pipelines.register_function(
    function=q2_mlab.preprocess,
    inputs={
        'table': FeatureTable[Frequency],
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
    ],
    input_descriptions={
        'table': 'The 16S or metagenomic feature table.',
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
    },
    name='Dataset preprocessing for benchmarking',
    description=('Applies filtering and preprocessing steps '
                 'to a feature table and metadata, and '
                 'generates distance matrices with phylo- '
                 'genetic and non-phylogenetic metrics.')
)

plugin.pipelines.register_function(
    function=q2_mlab.benchmark_classify,
    inputs={
        'table': FeatureTable[Frequency],
        'distance_matrix': DistanceMatrix,
        'metadata': SampleData[Target]
    },
    parameters={
        'param_index_start': Int % Range(1, None),
        'param_index_end': Int % Range(1, None),
        'n_jobs': Int % Range(-1, None),
    },
    outputs=[
        ('results_visualization', Visualization),
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
        'param_index_start': 'The index in the parameter list to begin '
                             'benchmarking from.',
        'param_index_end': 'The index in the parameter list to end '
                           'benchmarking at.',
        'n_jobs': '[beta methods only] - %s' % sklearn_n_jobs_description
    },
    output_descriptions={
        'results_visualization': 'Summary statistics',
    },
    name='Dataset preprocessing for benchmarking',
    description=('Applies filtering and preprocessing steps '
                 'to a feature table and metadata, and '
                 'generates distance matrices with phylo- '
                 'genetic and non-phylogenetic metrics.')
)
