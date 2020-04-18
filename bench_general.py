# coding: utf-8
#############The scikit-learn version is 0.19.1.
# In[25]:

# Import Libraries
import os
import math
import numpy as np
import pandas as pd
import scipy
import biom
import pickle
import time
import argparse
import calour as ca
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
from sklearn.model_selection import ParameterGrid
from skbio import DistanceMatrix
from scipy.sparse import *
from scipy.spatial import distance
from math import sqrt
from calour.training import RepeatedSortedStratifiedKFold


# In[26]:


# Import Regression Methods
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor


# In[31]:


parser = argparse.ArgumentParser()
parser.add_argument("dataset", 
                    help="the dataset you wish to benchmark with", 
                    type=int)
parser.add_argument('--linear', dest='linear', action='store_true')
parser.add_argument('--rf', dest='rf', action='store_true')
parser.add_argument('--rfd', dest='rfd', action='store_true')
parser.add_argument('--gb', dest='gb', action='store_true')
parser.add_argument('--gbd', dest='gbd', action='store_true')
parser.add_argument('--et', dest='et', action='store_true')
parser.add_argument('--mlp', dest='mlp', action='store_true')
parser.add_argument('--xgb', dest='xgb', action='store_true')
parser.add_argument('--xgbd', dest='xgbd', action='store_true')
parser.add_argument('--lsvr', dest='lsvr', action='store_true')
parser.add_argument('--knn', dest='knn', action='store_true')
parser.add_argument('--defaults', dest='run_defaults', action='store_true')
parser.add_argument('--use_default_grid', dest='use_default_grid', action='store_true')
parser.add_argument('--overwrite', dest='overwrite', action='store_true')

parser.set_defaults(linear=False)
parser.set_defaults(rf=False)
parser.set_defaults(rfd=False)
parser.set_defaults(gb=False)
parser.set_defaults(gbd=False)
parser.set_defaults(et=False)
parser.set_defaults(mlp=False)
parser.set_defaults(xgb=False)
parser.set_defaults(xgbd=False)
parser.set_defaults(lsvr=False)
parser.set_defaults(knn=False)
parser.set_defaults(default=False)
parser.set_defaults(use_default_grid=False)
parser.set_defaults(overwrite=False)

args = parser.parse_args()
dataset = args.dataset
knn = args.knn
run_defaults = args.run_defaults
use_default_grid = args.use_default_grid

# In[9]:


# Give a name for the output data file, directory prefixes
dir_prefixes = [
    '82-soil', #0
    'PMI_16s', #1
    'malnutrition', #2
    'cider', #3
    'oral_male', #4
    'oral_female', #5
    'skin_hand_female', #6
    'skin_hand_male', #7
    'skin_head_female', #8
    'skin_head_male', #9
    'gut_AGP_female', #10
    'gut_AGP_male', #11
    'gut_cantonese_female', #12
    'gut_cantonese_male', #13
    'soil_1883', #14
    'soil_2104', #15
    'soil_945', #16
    'soil_10442', #17
    'soil_10764', #18
    'soil_10082', #19
    'soil_1721', #20
    'soil_10251', #21
    'soil_10511', #22
    'soil_10278', #23
    'soil_1024', #24
    'soil_678', #25
    'soil_10470', #26
    'soil_755', #27
    'soil_905', #28
    'fermentation_2278', #29
    'fermentation_10119', #30
    'fermentation_1976_beer', #31
    'fermentation_1976_wine', #32
    "infant_fecal_11402", #33
    "infant_fecal_10918", #34
    "infant_fecal_10080", #35
    "infant_fecal_11358", #36
    "infant_oral_2010", #37
    "infant_skin_2010" #38
               ]

dir_prefix = dir_prefixes[dataset]
if not os.path.isdir(dir_prefix): 
        os.mkdir(dir_prefix, mode=0o755)

biom_fp = [
    '82-soil/rarefied_20000_filtered_samples_features_frequency_table.biom', #0
    'PMI_16s/rarefied_2000_filtered_samples_features_frequency_table.biom', #1
    'malnutrition/rarefied_8500_filtered_samples_features_frequency_table.biom', #2
    'cider/rarefied_2000_filtered_samples_features_frequency_table.biom',  #3
    "age_prediction/oral_4014/oral_4014__qiita_host_sex_female__.biom", #4
    "age_prediction/oral_4014/oral_4014__qiita_host_sex_male__.biom", #5
    "age_prediction/skin_4168/skin_4168__body_site_hand_qiita_host_sex_female__.biom",#6
    "age_prediction/skin_4168/skin_4168__body_site_hand_qiita_host_sex_male__.biom",#7
    "age_prediction/skin_4168/skin_4168__body_site_head_qiita_host_sex_female__.biom",#8
    "age_prediction/skin_4168/skin_4168__body_site_head_qiita_host_sex_male__.biom",#9
    "age_prediction/gut_4575/gut_4575_rare__cohort_AGP_sex_female__.biom",#10
    "age_prediction/gut_4575/gut_4575_rare__cohort_AGP_sex_male__.biom",#11
    "age_prediction/gut_4575/gut_4575_rare__cohort_cantonese_sex_female__.biom",#12
    "age_prediction/gut_4575/gut_4575_rare__cohort_cantonese_sex_male__.biom",#13
    'soil_1883/rarefied_2000_filtered_samples_features_frequency_table.biom', #14
    'soil_2104/rarefied_13000_filtered_samples_features_frequency_table.biom', #15
    'soil_945/rarefied_2000_filtered_samples_features_frequency_table.biom', #16
    'soil_10442/rarefied_6000_filtered_samples_features_frequency_table.biom', #17
    'soil_10764/rarefied_2000_filtered_samples_features_frequency_table.biom', #18
    'soil_10082/rarefied_4000_filtered_samples_features_frequency_table.biom', #19
    'soil_1721/rarefied_6000_filtered_samples_features_frequency_table.biom', #20
    'soil_10251/rarefied_20000_filtered_samples_features_frequency_table.biom', #21
    'soil_10511/rarefied_2000_filtered_samples_features_frequency_table.biom', #22
    'soil_10278/rarefied_6000_filtered_samples_features_frequency_table.biom', #23
    'soil_1024/rarefied_1000_filtered_samples_features_frequency_table.biom', #24
    'soil_678/rarefied_2000_filtered_samples_features_frequency_table.biom', #25
    'soil_10470/rarefied_8000_filtered_samples_features_frequency_table.biom', #26
    'soil_755/rarefied_20000_filtered_samples_features_frequency_table.biom', #27
    'soil_905/rarefied_14000_filtered_samples_features_frequency_table.biom', #28
    "fermentation_2278/rarefied_2000_filtered_samples_features_frequency_table.biom", #29
    "fermentation_10119/rarefied_1200_filtered_samples_features_frequency_table.biom", #30
    "fermentation_1976_beer/rarefied_4000_filtered_samples_features_frequency_table.biom", #31
    "fermentation_1976_wine/rarefied_2000_filtered_samples_features_frequency_table.biom", #32
    "infant_fecal_11402/rarefied_10000_filtered_samples_features_frequency_table.biom", #33
    "infant_fecal_10918/rarefied_9000_filtered_samples_features_frequency_table.biom", #34
    "infant_fecal_10080/rarefied_2000_filtered_samples_features_frequency_table.biom", #35
    "infant_fecal_11358/rarefied_8000_filtered_samples_features_frequency_table.biom", #36
    "infant_oral_2010/rarefied_2000_filtered_samples_features_frequency_table.biom", #37
    "infant_skin_2010/rarefied_2000_filtered_samples_features_frequency_table.biom" #38
              ]

metadata_fp = [
    '82-soil/20994_analysis_mapping_v3.tsv', 
    'PMI_16s/21159_analysis_mapping.txt',
    'malnutrition/merged_metadata_v3.txt',
    'cider/21291_analysis_mapping.txt',
    "age_prediction/oral_4014/oral_4014_map__qiita_host_sex_female__.txt",
    "age_prediction/oral_4014/oral_4014_map__qiita_host_sex_male__.txt",
    "age_prediction/skin_4168/skin_4168_map__body_site_hand_qiita_host_sex_female__.txt",
    "age_prediction/skin_4168/skin_4168_map__body_site_hand_qiita_host_sex_male__.txt",
    "age_prediction/skin_4168/skin_4168_map__body_site_head_qiita_host_sex_female__.txt",
    "age_prediction/skin_4168/skin_4168_map__body_site_head_qiita_host_sex_male__.txt",
    "age_prediction/gut_4575/gut_4575_rare_map__cohort_AGP_sex_female__FILTERED.txt", #10
    "age_prediction/gut_4575/gut_4575_rare_map__cohort_AGP_sex_male__FILTERED.txt",
    "age_prediction/gut_4575/gut_4575_rare_map__cohort_cantonese_sex_female__.txt",
    "age_prediction/gut_4575/gut_4575_rare_map__cohort_cantonese_sex_male__.txt"
              ]
metadata_fp = metadata_fp + ["/projects/ibm_aihl/adswafford/regression/20190408_pH_studies_benchmark/20190408_qiita_public_ph_150nt_metadata.txt"]*15
metadata_fp = metadata_fp + [
    "fermentation_2278/fermentation_2278.txt",
    "fermentation_10119/fermentation_10119.txt",
    "fermentation_1976_beer/qiita_v2.txt",
    "fermentation_1976_wine/qiita_v2.txt",
    "infant_fecal_11402/filtered_metadata.tsv",
    "infant_fecal_10918/filtered_metadata.tsv",
    "infant_fecal_10080/filtered_metadata.tsv",
    "infant_fecal_11358/filtered_metadata.tsv",
    "infant_oral_2010/filtered_metadata.tsv",
    "infant_skin_2010/filtered_metadata.tsv"
]


# In[10]:


if(knn): 
    exp = ca.read_amplicon(dir_prefixes[dataset]+"/feature-table.biom", metadata_fp[dataset], 
                       min_reads=None, normalize=None)
else:
    exp = ca.read_amplicon(biom_fp[dataset], metadata_fp[dataset], 
                       min_reads=None, normalize=None)
print(exp)


# In[11]:


target = None
#Specify column to predict
if (dataset==0): #82-soil
    target = 'ph'
if (dataset==1):
    target = 'days_since_placement'
if (dataset==2):
    target = 'Age_days'
if (dataset==3):
    target = 'fermentation_day'
if (dataset>=4 and dataset<10):
    target = 'qiita_host_age'
if (dataset>=10 and dataset<14):
    target = 'age'
if (dataset>=14 and dataset<29):
    target = 'ph'
if (dataset==29):
    target = 'stage_days'
if (dataset==30):
    target = 'stage'
if (dataset>=31 and dataset<33):
    target = 'time'
if (dataset>=33 and dataset<39):
    target = 'age' 
print("Target column: " + target)


# In[23]:


# Ensure no nan values in target variable for AGP/Cantonese datasets
if dataset >=4 and dataset <=13:
    meta_no_na = pd.read_csv(metadata_fp[dataset], "\t").dropna(subset=[target])
    table = biom.load_table(biom_fp[dataset])
    print("old: "+ str(table.shape[1]))
    table_no_na = table.filter(meta_no_na["#SampleID"])
    print("new: "+ str(table_no_na.matrix_data.T.shape[0]))
    
    exp.data = table_no_na.matrix_data.T
    exp.sample_metadata = meta_no_na


# In[22]:

# Ensure target is numeric
exp.sample_metadata[target] = pd.to_numeric(exp.sample_metadata[target])

# ## Modify parameter options by shape of data

# Create logarithmic scales for ranges of parameter options where valid inputs can be 1<->n_features or n_samples

# In[15]:


def get_logscale(end, num):
    scale = np.geomspace(start=1, stop=end-1, num=num)
    scale = list(np.around(scale, decimals=0).astype(int))
    return scale


# In[16]:

n_samples = exp.shape[0]
n_features = exp.shape[1]

#Logarithmic scales based on n_samples
s_logscale = get_logscale(n_samples, 11)
s_logscale7 = get_logscale(n_samples, 8)
s_logscale.pop()
s_logscale7.pop()

#Logarithmic scales based on n_features
f_logscale = get_logscale(n_features, 10)
f_logscale7 = get_logscale(n_features, 7)


# Why .pop()? Effective n_samples is less than total n_samples because one fold of samples is held out, so we remove the last item in the logscale. 
# ```
# ValueError: Expected n_neighbors <= n_samples,  but n_samples = 123, n_neighbors = 152
# ```

# In[29]:


# specify # processors in job script
cpu = -1

# KNeighbors for use with Distance Matrices
KNNDistance_grids = {'n_neighbors': s_logscale, 
             'weights':['uniform','distance'], 
             'algorithm': ['brute'],
             'n_jobs': [cpu],
             'p':[2],
             'metric':['precomputed'],
            } #20

# DecisionTree
DT_grids = {'criterion': ['mse'],
            'splitter': ['best','random'],
            'max_depth': s_logscale + [None], 
            'max_features': ['auto', 'sqrt', 'log2'],
            'random_state':[2018]
            } #66

# RandomForest
RF_grids = {'n_estimators': [1000],
            'criterion': ['mse'],
            'max_features': f_logscale + ['auto', 'sqrt', 'log2'], 
            'max_depth': s_logscale + [None], 
            'n_jobs': [cpu],
            'random_state': [2018],
            'bootstrap':[True,False],
            'min_samples_split': list(np.arange(0.01, 1, 0.2)),
            'min_samples_leaf': list(np.arange(0.01, .5, 0.1)) + [1],
           } #8580

# RandomForest, with fixed recommended parameters
# min_sample_split=0.01, min_sample_leaf=1, bootstrap=False
RF_rec_grids = {'n_estimators': [1000],
            'criterion': ['mse'],
            'max_features': f_logscale + ['auto', 'sqrt', 'log2'], 
            'max_depth': s_logscale + [None], 
            'n_jobs': [cpu],
            'random_state': [2018],
            'bootstrap':[False],
            'min_samples_split': [0.01],
            'min_samples_leaf': [1],
           } #143

# ExtraTrees
ET_grids = {'n_estimators': [1000],
            'criterion': ['mse'],
            'max_features': f_logscale + ['auto', 'sqrt', 'log2'], 
            'max_depth': s_logscale + [None], 
            'n_jobs': [cpu],
            'random_state': [2018],
            'bootstrap':[True,False],
            'min_samples_split': list(np.arange(0.01, 1, 0.2)),
            'min_samples_leaf': list(np.arange(0.01, .5, 0.1)) + [1],
           } #8580

# GradientBoosting
GB_grids = {'loss' : ['ls', 'lad', 'huber', 'quantile'],
            'alpha' : [1e-3, 1e-2, 1e-1, 0.5,0.9],
            'learning_rate': [3e-1, 2e-1, 1e-1, 5e-2],
            'n_estimators': [1000,5000],
            'criterion': ['mse'],
            'max_features': f_logscale7 + ['auto', 'sqrt', 'log2'], 
            'max_depth': s_logscale7 + [None], 
            'random_state': [2018]
            } #12800

GB_rec_grids = {'loss' : ['lad'],
            'alpha' : [1e-3, 1e-2, 1e-1, 0.5, 0.9],
            'learning_rate': [5e-2],
            'n_estimators': [5000],
            'criterion': ['mse'],
            'max_features': f_logscale7 + ['auto', 'sqrt', 'log2'], 
            'max_depth': s_logscale7 + [None], 
            'random_state': [2018]
            } #400

# Ridge
Ridge_grids = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,
                         1e-2, 1, 5, 10, 20],
               'fit_intercept': [True],
               'normalize': [True, False],
               'solver': ['auto', 'svd', 'cholesky', 'lsqr', 
                           'sparse_cg', 'sag', 'saga'],
               'random_state': [2018]
              } #140

# Lasso
Lasso_grids = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,
                         1e-2, 1, 5, 10, 20],
               'fit_intercept': [True],
               'normalize': [True, False],
               'random_state': [2018],
               'selection': ['random', 'cyclic']
              } #40

# ElasticNet
EN_grids = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,
                         1e-2, 1, 5, 10, 20],
            'l1_ratio': list(np.arange(0.0, 1.1, 0.1)),
            'fit_intercept': [True],
            'random_state': [2018],
            'selection': ['random', 'cyclic']
           } #200

# Linear SVR
LinearSVR_grids = {'C': [1e-4, 1e-3, 1e-2, 1e-1, 1e1, 
                   1e2, 1e3, 1e4, 1e5, 1e6, 1e7],
             'epsilon':[1e-2, 1e-1, 0, 1],
             'loss': ['squared_epsilon_insensitive', 'epsilon_insensitive'],
             'random_state': [2018]
            } #3520

# RBF SVR
RSVR_grids = {'C': [1e-4, 1e-3, 1e-2, 1e-1, 1e1, 
                   1e2, 1e3, 1e4, 1e5, 1e6, 1e7],
             'epsilon':[1e-4, 1e-3, 1e-2, 1e-5],
             'kernel':['rbf'],
             'gamma':['auto', 100, 10, 1, 1e-4, 1e-2, 1e-3, 
                      1e-4, 1e-5, 1e-6],
             'coef0':[0, 1, 10, 100]
            } #1760

# Sigmoid SVR
SSVR_grids = {'C': [1e-4, 1e-3, 1e-2, 1e-1, 1e1, 
                   1e2, 1e3, 1e4, 1e5, 1e6, 1e7],
             'epsilon':[1e-4, 1e-3, 1e-2, 1e-5],
             'kernel':['sigmoid'],
             'gamma':['auto', 100, 10, 1, 1e-4, 1e-2, 1e-3, 
                      1e-4, 1e-5, 1e-6],
             'coef0':[0, 1, 10, 100]
            }
            #'epsilon':[1e-2, 1e-1, 1e0, 1e1, 1e2]
            # Epsilon >10 causes divide by zero error, 
            # C<=0 causes ValueError: b'C <= 0'
            #1760

# XGBoost
XGB_grids = {'max_depth': s_logscale, ##########
             'learning_rate': [3e-1, 2e-1, 1e-1, 5e-2],
             'n_estimators': [1000,5000],
             'objective': ['reg:linear'],
             'booster': ['gbtree', 'gblinear'],
             'n_jobs': [cpu],
             'gamma': [0, 0.2, 0.5, 1, 3],
             'reg_alpha': [1e-3, 1e-1, 1],
             'reg_lambda': [1e-3, 1e-1, 1],
             'scale_pos_weight': [1],
             'base_score': [0.5],
             'random_state': [2018],
             'silent': [1] #no running messages will be printed
            } #9900

XGB_rec_grids = {'max_depth': s_logscale, ##########
             'learning_rate': [3e-1, 2e-1, 1e-1, 5e-2],
             'n_estimators': [5000],
             'objective': ['reg:linear'],
             'booster': ['gbtree'],
             'n_jobs': [cpu],
             'gamma': [0],
             'reg_alpha': [1e-3, 1e-1, 1],
             'reg_lambda': [1e-3, 1e-1, 1],
             'scale_pos_weight': [1],
             'base_score': [0.5],
             'random_state': [2018],
             'silent': [1] 
            } #360

# Multi-layer Perceptron 
MLP_grids = {'hidden_layer_sizes': [(100,),(200,),(100,50),(50,50),(25,25,25)],
             'activation': ['identity', 'logistic', 'tanh', 'relu'],
             'solver': ['lbfgs', 'sgd', 'adam'],
             'alpha': [1e-3, 1e-1, 1, 10, 100],
             'batch_size': ['auto'],
             'max_iter': [50,100,200,400],
             'learning_rate': ['constant'],
             'random_state': [2018,14,2362,3456,24,968,90],
            } #7,680

# Define the default parameter settings
defaults = dict()
defaults["RandomForest"] = {'n_estimators': [1000],
            'criterion': ['mse'],
            'max_features': ['auto'],
            'max_depth': [None], 
            'n_jobs': [-1],
            'random_state': [2018],
            'bootstrap': [True],
            'min_samples_split': [0.01],
            'min_samples_leaf':[1],
           }
defaults["GradientBoosting"] = {'loss' : ['ls'],
            'alpha' : [0.9],
            'learning_rate': [1e-1],
            'n_estimators': [1000],
            'criterion': ['mse'],
            'max_features': ['auto'],
            'max_depth': [3], ##########
            'random_state': [2018]
            }
defaults["ExtraTrees"]={'n_estimators': [1000],
            'criterion': ['mse'],
            'max_features': ['auto'],
            'max_depth': [None], 
            'n_jobs': [-1],
            'random_state': [2018],
            'bootstrap': [True],
            'min_samples_split': [0.01],
            'min_samples_leaf': [1],
           }
defaults["MLPRegressor"]={'hidden_layer_sizes': [(100,)],
             'activation': ['relu'],
             'solver': ['adam'],
             'alpha': [0.001], 
             'batch_size': ['auto'],
             'max_iter': [200],
             'learning_rate': ['constant'],
             'random_state': [2018]
            }
defaults["XGBRegressor"]={'max_depth':[3], ##########
             'learning_rate': [1e-1],
             'n_estimators': [1000],
             'objective': ['reg:linear'],
             'booster': ['gbtree'],
             'n_jobs': [-1],
             'gamma': [0],
             'reg_alpha': [1e-3],
             'reg_lambda': [1],
             'scale_pos_weight': [1],
             'base_score': [0.5],
             'random_state': [2018],
             'silent': [1]
            }
defaults["RadialSVR"]={'C': [1e1],
             'epsilon': [1e-2],
             'kernel': ['rbf'],
             'gamma': ['auto'],
             'coef0': [0]
            }
defaults["ElasticNet"]={'alpha': [1],
            'l1_ratio': [0.5],
            'fit_intercept': [True],
            'random_state': [2018],
            'selection': ['cyclic']
           }
defaults["Lasso"]={'alpha': [1],
               'fit_intercept': [True],
               'normalize': [False],
               'random_state': [2018],
               'selection': ['cyclic']
              }
defaults["DecisionTree"]={'criterion': ['mse'],
            'splitter': ['best'],
            'max_depth': [None],
            'max_features': ['auto'],
            'random_state': [2018]
            }
defaults["SigmoidSVR"]={'C': [1e1],
             'epsilon': [1e-2],
             'kernel': ['sigmoid'],
             'gamma': ['auto'],
             'coef0': [0]
            }
defaults["Ridge"]={'alpha': [1],
               'fit_intercept': [True],
               'normalize': [False],
               'solver': ['auto'],
               'random_state': [2018]
              }
defaults["LinearSVR"]={'C': [1e1],
             'epsilon': [1],
             'loss': ['epsilon_insensitive'],
             'random_state': [2018]
            }
defaults["KNN"]={'n_neighbors': [5], ##########
             'weights': ['uniform'], 
             'algorithm': ['brute'],
             'n_jobs': [-1],
             'metric': ['precomputed'],
             'p': [2]
            }

# In[18]:


reg_names = [
         "DecisionTree", 
         "RandomForest",
         "ExtraTrees", 
         "GradientBoosting",
         "Ridge", "Lasso", "ElasticNet", 
         "LinearSVR", "RadialSVR", "SigmoidSVR",
         "XGBRegressor",
         "MLPRegressor",
        ]
dm_names = [
    "jaccard",
    "aitchison",
    "weighted_unifrac",
    "unweighted_unifrac",
    "jensenshannon",
    ]

names = reg_names + dm_names 
dm_set = set(dm_names) 

# Each regressor and their grid, preserving order given above
regressors = [
    DecisionTreeRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    Ridge, Lasso, ElasticNet, 
    LinearSVR, SVR, SVR,
    XGBRegressor,
    MLPRegressor
]

regressors += [KNeighborsRegressor] * len(dm_names)  

all_param_grids = [
    DT_grids,
    RF_grids,
    ET_grids,
    GB_grids,
    Ridge_grids,
    Lasso_grids,
    EN_grids,
    LinearSVR_grids,
    RSVR_grids,
    SSVR_grids,
    XGB_grids,
    MLP_grids
]

all_param_grids += [KNNDistance_grids] * len(dm_names)

regFromName = dict(zip(names, regressors))
gridFromName = dict(zip(names, all_param_grids))


# In[19]:


# Modifiers if not running all models at once:
if (args.linear):
    names = [
         "DecisionTree", 
         "Ridge", "Lasso", "ElasticNet", "LinearSVR", 
         "RadialSVR", "SigmoidSVR"]
    regressors = [
                  DecisionTreeRegressor,
                  Ridge, Lasso, ElasticNet, LinearSVR,
                  SVR, SVR
                 ]
    all_param_grids = [
                       DT_grids,
                       Ridge_grids,
                       Lasso_grids,
                       EN_grids,
                       LinearSVR_grids,
                       RSVR_grids,
                       SSVR_grids
                      ]  
if (run_defaults):
    names = reg_names + dm_names
    all_param_grids = [defaults[name] for name in names] + [defaults["KNN"]] * len(dm_names)
    
if (args.knn):
    names = dm_names
    regressors = [KNeighborsRegressor] * len(dm_names)
    all_param_grids = [KNNDistance_grids] * len(dm_names)
    if use_default_grid:
        all_param_grids = [defaults["KNN"]]
        
if (args.rf):
    names = ["RandomForest"]
    regressors = [RandomForestRegressor]
    all_param_grids = [RF_grids]
    if use_default_grid:
        all_param_grids = [defaults["RandomForest"]]
        
if (args.rfd):
    names = ["RandomForest_rec"]
    regressors = [RandomForestRegressor]
    all_param_grids = [RF_rec_grids]
    
if (args.et):
    names = ["ExtraTrees"]
    regressors = [ExtraTreesRegressor]
    all_param_grids = [RF_grids]
    if use_default_grid:
        all_param_grids = [defaults["ExtraTrees"]]
        
if (args.gb):    
    names = ["GradientBoosting"]
    regressors = [GradientBoostingRegressor]
    all_param_grids = [GB_grids]
    if use_default_grid:
        all_param_grids = [defaults["GradientBoosting"]]
        
if (args.gbd):
    names = ["GradientBoosting_rec"]
    regressors = [GradientBoostingRegressor]
    all_param_grids = [GB_rec_grids]
    
if (args.mlp):
    names = ["MLPRegressor"]
    regressors = [MLPRegressor]
    all_param_grids = [MLP_grids]
    if use_default_grid:
        all_param_grids = [defaults["MLPRegressor"]]
        
if (args.xgb):
    names = ["XGBRegressor"]
    regressors = [XGBRegressor]
    all_param_grids = [XGB_grids]
    if use_default_grid:
        all_param_grids = [defaults["XGBRegressor"]]
        
if (args.xgbd):
    names = ["XGBRegressor_rec"]
    regressors = [XGBRegressor]
    all_param_grids = [XGB_rec_grids]
    
if (args.lsvr):
    names = ["LinearSVR"]
    regressors = [LinearSVR]
    all_param_grids = [LinearSVR_grids]
    if use_default_grid:
        all_param_grids = [defaults["LinearSVR"]]


# ## Main benchmarking loop

for reg_idx, (reg, name, grid) in enumerate(zip(regressors, names, all_param_grids)):
    
    if (run_defaults):
        print("Running default parameters for " + name)
    
    is_distmatrix = name in dm_set #Boolean switch for distance-matrix specific code blocks
    if is_distmatrix: ##### Use specific X and y for distance matrix benchmarking, not amplicon experiment object
        
        if name=="jensenshannon":
            md = exp.sample_metadata
            existing_dm = DistanceMatrix.read(dir_prefix+"/beta-q2/"+"aitchison"+'.txt')
            print("Computing Jensen-Shannon Distance Matrix")
            dm = DistanceMatrix(data=distance.pdist(exp.data.todense(), metric="jensenshannon"), ids=existing_dm.ids)
        else:
            dm = DistanceMatrix.read(dir_prefix+"/beta-q2/"+name+'.txt')
            
        md = exp.sample_metadata
        md = md.filter(dm.ids,axis='index')
        dm = dm.filter(md.index, strict=True)
        
        X_dist = dm.data
        y_dist = md[target]
        
    # Make directory for this regressor if it does not yet exist
    dir_name = dir_prefix +'/' +dir_prefix + '-' + name
    print(dir_name)
    if not os.path.isdir(dir_name): 
        os.mkdir(dir_name, mode=0o755)

    paramsList = list(ParameterGrid(grid))   
       
    # For each set of parameters, get scores for model across 10 folds
    for param_idx, param in enumerate(paramsList):
        
        if (run_defaults): 
                param_idx = "Default"
        
        # If the benchmark data for this param set does not yet exist, benchmark it
        results_exist = os.path.isfile(dir_name+'/'+str(param_idx).zfill(5)+'_predictions.pkl')
        if ((not results_exist) or args.overwrite):
            print(str(param_idx) + " Starting: ")
            print(param)
            if is_distmatrix: #If benchmarking distance matrix:
                
                # new splits generator for each set of parameterzs
                if (dataset==2): #Use GroupKFold with Malnutrition dataset (2)
                    splits = GroupKFold(n_splits = 16).split(X = X_dist, y = y_dist, 
                                                             groups = md['Child_ID'])
                else:
                    splits = RepeatedSortedStratifiedKFold(5, 3, random_state=2018).split(X_dist, y_dist)
                    
                ### Start Timing
                start = time.process_time()
                df = pd.DataFrame(columns = ['CV', 'SAMPLE', 'Y_PRED', 'Y_TRUE'])
                cv_idx = 0
                
                CV = []
                Y_PRED = []
                Y_TRUE = []
                Y_IDS = []
                
                for train_index, test_index in splits: #y_classes
                    if is_distmatrix:
                        X_train, X_test = X_dist[train_index,:][:,train_index], X_dist[list(test_index),:][:,list(train_index)]
                    else:
                        X_train, X_test = X_dist[train_index], X_dist[test_index]
                    y_train, y_test = y_dist[train_index], y_dist[test_index]
                    y_train = np.asarray(y_train, dtype='int')
                    y_test_ids = y_dist.index[test_index] ####
                    
                    if is_distmatrix:
                        m = KNeighborsRegressor(**param)
                    else:
                        m = reg(**param)

                    m.fit(X_train, y_train)
                    y_pred = m.predict(X_test)

                    CV.extend([cv_idx] * len(y_pred))
                    Y_PRED.extend(y_pred)
                    Y_TRUE.extend(y_test)
                    Y_IDS.extend(y_test_ids)
                    
                    cv_idx += 1
                  
                df['CV'] = CV
                df['Y_TRUE'] = Y_TRUE
                df['Y_PRED'] = Y_PRED
                df['SAMPLE'] = Y_IDS
                end = time.process_time() 
                
                ### End Timing
            
            else: #All others; Not benchmarking distance matrix
            
                if (dataset==2): #Use GroupKFold with Malnutrition dataset (2)
                    it = exp.regress(target, reg(),
                                     cv = GroupKFold(n_splits = 16).split(X = exp.data, 
                                                                          y = exp.sample_metadata['Age_days'], 
                                                                          groups = exp.sample_metadata['Child_ID']), 
                                     params=[param])
                else:
                    it = exp.regress(target, reg(),
                                     cv = RepeatedSortedStratifiedKFold(5, 3, random_state=2018),
                                     params=[param])

                ### Start Timing
                start = time.process_time()
                df = next(it)
                end = time.process_time()
                ### End Timing
            
            #### Predictions-level dataframe, saved by param_idx
            df.to_pickle(dir_name+'/'+str(param_idx).zfill(5)+'_predictions.pkl') 
            
            
            #### Create folds-level dataframe
            # Check for NaN in x['Y_PRED'].values:  
            contains_nan = False
            if (np.any(pd.isnull(df['Y_PRED'].values))) or  (not np.all(np.isfinite(df['Y_PRED'].values))):
                print(df['Y_PRED'].values)
                print("Parameter set contains infinity, param_idx: " + str(param_idx))
                contains_nan = True           
            # Calculate RMSE for each fold in this set
            fold_rmse = pd.DataFrame()
            if contains_nan:
                fold_rmse['RMSE'] = df.groupby('CV').apply(lambda x: np.sqrt(x['Y_TRUE'].values))
                fold_rmse['RMSE'] = [None] * fold_rmse.shape[0]
            else:
                fold_rmse['RMSE'] = df.groupby('CV').apply(lambda x: np.sqrt(mean_squared_error(x['Y_PRED'].values, x['Y_TRUE'].values)))
            fold_rmse['PARAM'] = [param] * fold_rmse.shape[0]
   
            # Store runtimes for this param set
            param_runtime = end-start
            fold_rmse['RUNTIME'] = [param_runtime] * fold_rmse.shape[0]
            fold_rmse.to_pickle(dir_name+'/'+str(param_idx).zfill(5)+'_fold_rmse.pkl')
            


print("End")


# ## NULL MODELS
# * Needs one null model per dataset
# * Randomly permute y_true 100 times, and compare each permutation to y_true (RMSE)
# * Series, len=102, looks like [mean, median, RMSE_00, ... RMSE99]
# * Saved to pkl 

# In[34]:


y_true = exp.sample_metadata[target].values
data = []
index = []

for i in range(0,100):
    index.append('RMSE_'+str(i))
    y_perm = np.random.permutation(y_true)
    data.append(sqrt(mean_squared_error(y_perm, y_true)))
    
data = [np.mean(data), np.median(data)] + data
index = ['MEAN', "MEDIAN"] + index
null_model = pd.Series(data, index)

null_model.to_pickle("NULL_MODEL_"+dir_prefix+".pkl")

