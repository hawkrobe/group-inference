##### import packages
#base
import os
from collections import defaultdict
import numpy as np
import scipy.stats
from matplotlib import pyplot as plt
import random
import pyreadr
import numpy as np
import pandas as pd
import seaborn as sns
# %matplotlib inline
# %autosave 30

#pyro contingency
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.autoguide import AutoDelta
from pyro.optim import Adam
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete, Predictive
from pyro.ops.indexing import Vindex
from pyro.infer import MCMC, NUTS
import torch
from torch.distributions import constraints
pyro.enable_validation(True)

# import umap
# import plotly
# import plotly.graph_objs as go

#misc
import pickle
import torch.nn.functional as F
import itertools
import time

import arviz as az

# import homebrew modules
import tomtom_models as tm
import tomtom_util as tu

########## import data
# import data and previously fitted parameters
# import pickled data
with open('tomtom_data_preprocessed.pkl','rb') as f:
    [tself_norm_all_3d, tself_norm_noauto_3d, tself_raw_all_3d, tself_raw_noauto_3d,
    ttarg_norm_all_3d, ttarg_norm_noauto_3d, ttarg_raw_all_3d, ttarg_raw_noauto_3d,
    tavg_norm_all_3d, tavg_norm_noauto_3d, tavg_raw_all_3d, tavg_raw_noauto_3d] = pickle.load(f)
# import pickled parameters from varying-k analysis
with open('tomtom_fitted_models.pkl','rb') as f:
    [seeds_self_norm_all_grp,maps_self_norm_all_grp,logprobs_self_norm_all_grp,mems_self_norm_all_grp,
     seeds_self_norm_all_dim,maps_self_norm_all_dim,logprobs_self_norm_all_dim,
     seeds_self_norm_noauto_grp,maps_self_norm_noauto_grp,logprobs_self_norm_noauto_grp,mems_self_norm_noauto_grp,
     seeds_self_norm_noauto_dim,maps_self_norm_noauto_dim,logprobs_self_norm_noauto_dim,
     seeds_self_raw_all_grp,maps_self_raw_all_grp,logprobs_self_raw_all_grp,mems_self_raw_all_grp,
     seeds_self_raw_all_dim,maps_self_raw_all_dim,logprobs_self_raw_all_dim,
     seeds_self_raw_noauto_grp,maps_self_raw_noauto_grp,logprobs_self_raw_noauto_grp,mems_self_raw_noauto_grp,
     seeds_self_raw_noauto_dim,maps_self_raw_noauto_dim,logprobs_self_raw_noauto_dim] = pickle.load(f)

# # load previously generated data
# with open('model_recovery_gen_dat.pkl','rb') as f:
#     [gendat_self_norm_all_grp,gendat_self_norm_all_dim,
#     gendat_self_raw_noauto_grp,gendat_self_raw_noauto_dim] = pickle.load(f)

# load previously generated smaller set of data
with open('model_recovery_gen_dat_small.pkl','rb') as f:
    [gendat_self_norm_all_grp,gendat_self_norm_all_dim,
    gendat_self_raw_noauto_grp,gendat_self_raw_noauto_dim,
    gendat_self_norm_noauto_grp,gendat_self_norm_noauto_dim,
    gendat_self_raw_all_grp,gendat_self_raw_all_dim] = pickle.load(f)
print('Read in small dataset!')

# code to generate data from params
# define separate functions for group model and dimension model
#  group
def datagen_grp(n_sample, map_est):
    # initialize storage
    stor = []
    if 'concentration' in map_est.keys():
        for i in np.arange(n_sample):
            grp = dist.Categorical(map_est['weights']).sample()
            grp_c = map_est['concentration'][grp]
            samp = dist.Dirichlet(grp_c).sample()
            stor.append(samp)
    else:
        for i in np.arange(n_sample):
            grp = dist.Categorical(map_est['weights']).sample()
            grp_a = map_est['alpha'][grp]
            grp_b = map_est['beta'][grp]
            samp = dist.Beta(grp_a, grp_b).sample()
            stor.append(samp)
    return torch.stack(stor)

# # short function to add N empty dimensions to a tensor
# def add_n_dim(t,n):
#     for i in np.arange(n):
#         t = t[:,None]
#     return t

#  dimension
def datagen_dim(n_sample, map_est):
    stor = []
    if 'topic_concentration' in map_est.keys():
        for i in np.arange(n_sample):
            # sample participant level topic weights
            p_w = dist.Dirichlet(map_est['topic_weights']).sample()
            # component matrices
            c_m = dist.Dirichlet(map_est['topic_concentration']).sample()
            # normalize components with participant weights
            c_m_weighted = p_w[:,None,None] * c_m # make broadcasting work
            weighted_mixture = c_m_weighted.sum(0)
            stor.append(weighted_mixture)
    else:
        for i in np.arange(n_sample):
            # sample participant level topic weights
            p_w = dist.Dirichlet(map_est['topic_weights']).sample()
            # component matrices
            c_m = dist.Beta(map_est['topic_a'], map_est['topic_b']).sample()
            # normalize components with participant weights
            c_m_weighted = p_w[:,None,None] * c_m # make broadcasting work
            weighted_mixture = c_m_weighted.sum(0)
            stor.append(weighted_mixture)
    return torch.stack(stor)

# actually generating new data
## varying along: K, n_sample
## for each combo of the two repeat 5
n_repeat = 5
n_sample_array = np.concatenate((np.arange(10,160,15), np.arange(200,600,100)))
# n_sample_array = np.arange(10,30,10)
# wrap in a function so this can be done for maps from different fitted model params
# takes in an array of MAPs, model type ('grp'/'dim'),n repeat, all values of n_sample
# return nested list
    # l1 organized by K (index+1)
    # l2 organize by n_sample_array
    # l3 are tensors: n repeat * n_sample * data_dimensions
def datagen_combo(maps, mtype, n_repeat, n_sample_array):
    stor1 = []
    for mmap in maps:
        stor2 = []
        for n_sample in n_sample_array:
            stor3 = []
            for i in np.arange(n_repeat):
                if mtype == 'grp':
                    dat = datagen_grp(n_sample, mmap)
                elif mtype == 'dim':
                    dat = datagen_dim(n_sample, mmap)
#                 print(dat.shape)
                stor3.append(dat)
#             print(torch.stack(stor3).shape)
            stor2.append(torch.stack(stor3))
        stor1.append(stor2)
    return stor1

# random.seed(20201103)
# gendat_self_norm_all_grp = datagen_combo(maps_self_norm_all_grp, 'grp', n_repeat, n_sample_array)
# gendat_self_norm_all_dim = datagen_combo(maps_self_norm_all_dim, 'dim', n_repeat, n_sample_array)
# gendat_self_raw_noauto_grp = datagen_combo(maps_self_raw_noauto_grp, 'grp', n_repeat, n_sample_array)
# gendat_self_raw_noauto_dim = datagen_combo(maps_self_raw_noauto_dim, 'dim', n_repeat, n_sample_array)
#
# # pickle generated data
# with open('model_recovery_gen_dat_small.pkl','wb') as f:
#     pickle.dump([gendat_self_norm_all_grp,gendat_self_norm_all_dim,
#                 gendat_self_raw_noauto_grp,gendat_self_raw_noauto_dim],f)
# print('done with new small set data generation!')

## define function to detach parameter estimates so results can be sent back from multiprocessiong
def detach_mmap(mmap):
    for k in mmap.keys():
        mmap[k] = mmap[k].detach()
    return mmap

# ### defining function that iteratively refit the model
# def tomtom_refit(gendat, print_fit = False):
#     modrec_seeds = []
#     modrec_maps = []
#     modrec_logprobs = []
#     tm.K = 1
#     for kmap in gendat:
#         print('Currently refitting {} model with K={}'.format(tm.mtype, tm.K))
#         # each element in the second layer is a tensor nrep*nsample*datadim
#         stor1_seeds = []
#         stor1_maps = []
#         stor1_logprobs = []
#         for tn in np.arange(len(kmap)):
#             print('Where sample size is {}'.format(n_sample_array[tn]))
#             # iterate through the first dimension of the tensor, fitting model for each layer
#             stor2_seeds = []
#             stor2_maps = []
#             stor2_logprobs = []
#             tens = kmap[tn]
#             for i in np.arange(tens.shape[0]):
#                 if 'gr' in tm.mtype:
#                     seed, mmap, mem, lp = tm.tomtom_svi(tens[i], print_fit = print_fit)
#                 elif 'di' in tm.mtype:
#                     seed, mmap, lp = tm.tomtom_svi(tens[i], print_fit = print_fit)
#                 stor2_seeds.append(seed)
#                 stor2_maps.append(mmap)
#                 stor2_logprobs.append(lp)
#             stor1_seeds.append(stor2_seeds)
#             stor1_maps.append(stor2_maps)
#             stor1_logprobs.append(stor2_logprobs)
#         modrec_seeds.append(stor1_seeds)
#         modrec_maps.append(stor1_maps)
#         modrec_logprobs.append(stor1_logprobs)
#         tm.K += 1
#     return modrec_seeds, modrec_maps, modrec_logprobs

## redefining tomtom_refit for multiprocessiong
def tomtom_refit(zipkmap):
    tm.K = zipkmap[0]
    kmap = zipkmap[1]
    print('Currently refitting {} model with K={}'.format(tm.mtype, tm.K),flush=True)
    # each element in the second layer is a tensor nrep*nsample*datadim
    stor1_seeds = []
    stor1_maps = []
    stor1_logprobs = []
    for tn in np.arange(len(kmap)):
        print('Where K = {} and sample size is {}'.format(tm.K, n_sample_array[tn]),flush=True)
        # iterate through the first dimension of the tensor, fitting model for each layer
        stor2_seeds = []
        stor2_maps = []
        stor2_logprobs = []
        tens = kmap[tn]
        for i in np.arange(tens.shape[0]):
            if 'gr' in tm.mtype:
                seed, mmap, mem, lp = tm.tomtom_svi(tens[i], print_fit = False)
            elif 'di' in tm.mtype:
                seed, mmap, lp = tm.tomtom_svi(tens[i], print_fit = False)
            stor2_seeds.append(seed)
            stor2_maps.append(detach_mmap(mmap))
            stor2_logprobs.append(lp.detach())
        stor1_seeds.append(stor2_seeds)
        stor1_maps.append(stor2_maps)
        stor1_logprobs.append(stor2_logprobs)
    return stor1_seeds,stor1_maps,stor1_logprobs


# norm all dim
tm.mtype = 'group'
tm.target = 'self' # 'self','targ','avg'
tm.dtype = 'raw' # 'norm','raw'
tm.auto = 'noauto' # 'noauto','all'
tm.stickbreak = False
tm.optim = pyro.optim.Adam({'lr': 0.0005, 'betas': [0.8, 0.99]})
tm.elbo = TraceEnum_ELBO(max_plate_nesting=1)

# modrec_seeds_self_raw_noauto_grp, modrec_maps_self_raw_noauto_grp, modrec_logprobs_self_raw_noauto_grp = tomtom_refit(gendat_self_raw_noauto_grp)
import multiprocessing
# zip each element of gendat with its associated K for ease of pooling
def zip_gendat(gendat):
    return [(i+1, gendat[i]) for i in range(len(gendat))]
gendat = zip_gendat(gendat_self_raw_all_grp)
# pooling
pool = multiprocessing.Pool()
poolout = pool.map(tomtom_refit,gendat)
modrec_seeds_self_raw_all_grp = [i[0] for i in poolout]
modrec_maps_self_raw_all_grp = [i[1] for i in poolout]
modrec_logprobs_self_raw_all_grp = [i[2] for i in poolout]

# save
with open('modrec_self_raw_all_grp.pkl','wb') as f:
    pickle.dump([modrec_seeds_self_raw_all_grp,modrec_maps_self_raw_all_grp,modrec_logprobs_self_raw_all_grp],f)
