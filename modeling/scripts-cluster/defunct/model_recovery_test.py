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
    [seeds_self_norm_all_grp,maps_self_norm_all_grp,logprobs_self_norm_all_grp,mem_self_norm_all_grp,
     seeds_self_norm_all_dim,maps_self_norm_all_dim,logprobs_self_norm_all_dim,
     seeds_self_raw_noauto_grp,maps_self_raw_noauto_grp,logprobs_self_raw_noauto_grp,mem_self_raw_noauto_grp,
     seeds_self_raw_noauto_dim,maps_self_raw_noauto_dim,logprobs_self_raw_noauto_dim] = pickle.load(f)

# load previously generated data
with open('model_recovery_gen_dat.pkl','rb') as f:
    [gendat_self_norm_all_grp,gendat_self_norm_all_dim,
    gendat_self_raw_noauto_grp,gendat_self_raw_noauto_dim] = pickle.load(f)

# norm all grp
# Fitting a priori K GROUP model on normed all data
tm.mtype = 'group'
tm.target = 'self' # 'self','targ','avg'
tm.dtype = 'norm' # 'norm','raw'
tm.auto = 'all' # 'noauto','all'
tm.stickbreak = False
tm.optim = pyro.optim.Adam({'lr': 0.0005, 'betas': [0.8, 0.99]})
tm.elbo = TraceEnum_ELBO(max_plate_nesting=1)

# initializing storage
modrec_seeds_self_norm_all_grp = []
modrec_maps_self_norm_all_grp = []
modrec_logprobs_self_norm_all_grp = []
# each element in the outermost list is all gendat for a singel k-MAP
tm.K = 1
for kmap in gendat_self_norm_all_grp:
    print(tm.K)
    # each element in the second layer is a tensor nrep*nsample*datadim
    stor1_seeds_self_norm_all_grp = []
    stor1_maps_self_norm_all_grp = []
    stor1_logprobs_self_norm_all_grp = []
    for tens in kmap:
        # iterate through the first dimension of the tensor, fitting model for each layer
        stor2_seeds_self_norm_all_grp = []
        stor2_maps_self_norm_all_grp = []
        stor2_logprobs_self_norm_all_grp = []
        for i in np.arange(tens.shape[0]):
            pyro.clear_param_store()
            seed, mmap, mem, lp = tm.tomtom_svi(tens[i])
            stor2_seeds_self_norm_all_grp.append(seed)
            stor2_maps_self_norm_all_grp.append(mmap)
            stor2_logprobs_self_norm_all_grp.append(lp)
        stor1_seeds_self_norm_all_grp.append(stor2_seeds_self_norm_all_grp)
        stor1_maps_self_norm_all_grp.append(stor2_maps_self_norm_all_grp)
        stor1_logprobs_self_norm_all_grp.append(stor2_logprobs_self_norm_all_grp)
    modrec_seeds_self_norm_all_grp.append(stor1_seeds_self_norm_all_grp)
    modrec_maps_self_norm_all_grp.append(stor1_maps_self_norm_all_grp)
    modrec_logprobs_self_norm_all_grp.append(stor1_logprobs_self_norm_all_grp)
    tm.K += 1

with open('refit_mod_self_norm_all_grp','wb') as f:
    pickle.dump([modrec_maps_self_norm_all_grp,modrec_seeds_self_norm_all_grp,modrec_logprobs_self_norm_all_grp],f)
