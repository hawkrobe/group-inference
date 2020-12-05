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

# import data
# global tself_norm_all_3d, tself_norm_noauto_3d, tself_raw_all_3d, tself_raw_noauto_3d
# global ttarg_norm_all_3d, ttarg_norm_noauto_3d, ttarg_raw_all_3d, ttarg_raw_noauto_3d
# global tavg_norm_all_3d, tavg_norm_noauto_3d, tavg_raw_all_3d, tavg_raw_noauto_3d

# import pickled data
with open('tomtom_data_preprocessed.pkl','rb') as f:
    [tself_norm_all_3d, tself_norm_noauto_3d, tself_raw_all_3d, tself_raw_noauto_3d,
    ttarg_norm_all_3d, ttarg_norm_noauto_3d, ttarg_raw_all_3d, ttarg_raw_noauto_3d,
    tavg_norm_all_3d, tavg_norm_noauto_3d, tavg_raw_all_3d, tavg_raw_noauto_3d] = pickle.load(f)

# set model fitting params
tm.K = 3
tm.mtype = 'group'
tm.target = 'self' # 'self','targ','avg'
tm.dtype = 'raw' # 'norm','raw'
tm.auto = 'noauto' # 'noauto','all'
tm.stickbreak = False
tm.optim = pyro.optim.Adam({'lr': 0.0005, 'betas': [0.8, 0.99]})
tm.elbo = TraceEnum_ELBO(max_plate_nesting=1)
dtname = 't{}_{}_{}_3d'.format(tm.target, tm.dtype, tm.auto)
data = globals()[dtname]
seed_grp, mapl_grp, mem_grp, lp_grp, guide_grp = tm.tomtom_svi(data,return_guide = True)

# defining sparse-input model
@config_enumerate
def model_multi_obs_grp(obsmat):
    # some parameters can be directly derived from the data passed
    # K = 2
    nparticipants = data.shape[0]
    nfeatures = data.shape[1] # number of rows in each person's matrix
    ncol = data.shape[2]

    # Background probability of different groups
    if tm.stickbreak:
        # stick breaking process for assigning weights to groups
        with pyro.plate("beta_plate", K-1):
            beta_mix = pyro.sample("weights", dist.Beta(1, 10))
        weights = tm.mix_weights(beta_mix)
    else:
        weights = pyro.sample('weights', dist.Dirichlet(0.5 * torch.ones(tm.K)))
    # declare model parameters based on whether the data are row-normalized
    if tm.dtype == 'norm':
        pass
#         with pyro.plate('components', K):
#             # concentration parameters
#             concentration = pyro.sample('concentration',
#                                         dist.Gamma(2 * torch.ones(nfeatures,ncol), 1/3 * torch.ones(nfeatures,ncol)).to_event(2))

#         # implementation for the dirichlet based model is not complete!!!!
#         with pyro.plat('data',obsmat.shape[0]):
#             assignment = pyro.sample('assignment', dist.Categorical(weights))
#             #d = dist.Dirichlet(concentration[assignment,:,:].clone().detach()) # .detach() might interfere with backprop
#             d = dist.Dirichlet(concentration[assignment,i,:])
#             pyro.sample('obs', d.to_event(1), obs=obsmat)

    elif tm.dtype == 'raw':
        with pyro.plate('components', tm.K):
            alphas = pyro.sample('alpha', dist.Gamma(2 * torch.ones(nfeatures,ncol), 1/3 * torch.ones(nfeatures,ncol)).to_event(2))
            betas = pyro.sample('beta', dist.Gamma(2 * torch.ones(nfeatures,ncol), 1/3 * torch.ones(nfeatures,ncol)).to_event(2))

        assignment = pyro.sample('assignment', dist.Categorical(weights))
        # expand assignment to make dimensions match
        for r in np.arange(obsmat.shape[0]):
            rowind = obsmat[r,1].type(torch.long)
            colind = obsmat[r,2].type(torch.long)
            d = dist.Beta(alphas[assignment,rowind,colind],betas[assignment,rowind,colind])
            pyro.sample('obs_{}'.format(r), d, obs = obsmat[r,0])

# multi_obs classifier
guide_trace = poutine.trace(guide_grp).get_trace(data)  # record the globals
trained_model_multi = poutine.replay(model_multi_obs_grp, trace = guide_trace)

def classifier_multi_obs(obsmat, temperature): # temperature = 1 to sample
    inferred_model = infer_discrete(trained_model_multi, temperature=temperature,
                                    first_available_dim=-1)  # avoid conflict with data plate
    trace = poutine.trace(inferred_model).get_trace(obsmat)
    return trace.nodes["assignment"]["value"]

# Single Feature
# initialize storage
niter = 500
nsteps = np.arange(.1,1,.1).shape[0]
stor_grp = torch.empty(size = [data.shape[1],data.shape[2],nsteps])
stor_grp_prb = torch.empty(size = [data.shape[1],data.shape[2],nsteps,tm.K])
# iterate through all features, each with value .1-.9
for row in np.arange(data.shape[1]):
    for col in np.arange(data.shape[2]):
        step_count = 0
        for step in np.arange(.1,1,.1):
            newdata = torch.tensor([step, row, col]).unsqueeze(0)
            # first MAP classification
            grp = classifier_multi_obs(newdata, temperature = 0)
            stor_grp[row,col,step_count] = grp
            # second use sampling to get group prob
            stor = torch.zeros(niter)
            for it in np.arange(niter):
                stor[it] = classifier_multi_obs(newdata, temperature = 1)
            grp_prb = [(stor == i).sum()/float(len(stor)) for i in np.arange(tm.K)]
            stor_grp_prb[row,col,step_count,:] = torch.tensor(grp_prb)
            step_count += 1

# quick check
print(stor_grp[0])
print(stor_grp_prb[0])

# store single feature output
with open('stor_grp_1feat.pkl','wb') as f:
    pickle.dump([stor_grp, stor_grp_prb],f)
