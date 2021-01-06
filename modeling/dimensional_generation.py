
##### import packages
#base
import os
import sys
sys.path.append('../../')
from collections import defaultdict
import numpy as np
import scipy.stats
from matplotlib import pyplot as plt
import random
import pandas as pd
import seaborn as sns

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

#misc
import pickle
import torch.nn.functional as F
import itertools
import time

# import homebrew modules
import models.tomtom_models as tm
import models.tomtom_util as tu

# some useless warnings from seaborn, suppressing here
import warnings
warnings.filterwarnings("ignore")

with open('../data/tomtom_data_preprocessed.pkl','rb') as f:
    [tself_norm_all_3d, tself_norm_noauto_3d, tself_raw_all_3d, tself_raw_noauto_3d,
    ttarg_norm_all_3d, ttarg_norm_noauto_3d, ttarg_raw_all_3d, ttarg_raw_noauto_3d,
    tavg_norm_all_3d, tavg_norm_noauto_3d, tavg_raw_all_3d, tavg_raw_noauto_3d] = pickle.load(f)
    
tm.K = 3
tm.mtype = 'dim'
tm.target = 'self' # 'self','targ','avg'
tm.dtype = 'raw' # 'norm','raw'
tm.auto = 'noauto' # 'noauto','all'
tm.stickbreak = False
tm.optim = pyro.optim.Adam({'lr': 0.001, 'betas': [0.8, 0.99]})
tm.elbo = TraceEnum_ELBO(max_plate_nesting=1)

dtname = 't{}_{}_{}_3d'.format(tm.target, tm.dtype, tm.auto)
data = globals()[dtname]
seed_dim, mapl_dim, lp_dim, guide_dim = tm.tomtom_svi(data,return_guide = True)

# store a copy of MAP without participant topics for new inference purposes
map_dim_nopt = mapl_dim.copy()
pt = map_dim_nopt.pop('participant_topics')

# scarse obs model and guide
@config_enumerate
def model_multi_obs_dim(obsmat):
    num_topics = tm.K
    nparticipants = data.shape[0]
    nfeatures = data.shape[1] # number of rows in each person's matrix
    ncol = data.shape[2]            
    with pyro.plate('topic', num_topics):
        # sample a weight and value for each topic
        topic_weights = pyro.sample("topic_weights", dist.Gamma(1. / num_topics, 1.))
        topic_a = pyro.sample(
            "topic_a",
            dist.Gamma(2 * torch.ones(nfeatures, ncol),
                       1/3 * torch.ones(nfeatures, ncol)).to_event(2)
        )
        topic_b = pyro.sample(
            "topic_b",
            dist.Gamma(2 * torch.ones(nfeatures, ncol),
                       1/3 * torch.ones(nfeatures, ncol)).to_event(2)
        )
        
    # sample each participant's idiosyncratic topic mixture
    participant_topics = pyro.sample("new_participant_topic", dist.Dirichlet(topic_weights))
    transition_topics = pyro.sample("new_transition_topic", dist.Categorical(participant_topics),
                                    infer={"enumerate": "parallel"})
    rowind = obsmat[1].type(torch.long)
    colind = obsmat[2].type(torch.long)
    d = dist.Beta(topic_a[transition_topics, rowind, colind],
                  topic_b[transition_topics, rowind, colind])
    pyro.sample('obs', d, obs = obsmat[0])
          
@config_enumerate
def new_guide(obsmat):
    # Global variables.
    initial_topic_weights = pyro.get_param_store()['AutoDelta.topic_weights']
    initial_alpha = pyro.get_param_store()['AutoDelta.topic_weights']
    initial_topic_a = pyro.get_param_store()['AutoDelta.topic_a']
    initial_topic_b = pyro.get_param_store()['AutoDelta.topic_b']
    with poutine.block(hide_types=["param"]):  # Keep our learned values of global parameters.
        with pyro.plate('topic', tm.K):
            pyro.sample("topic_weights", dist.Delta(
                pyro.param('AutoDelta.topic_weights', initial_topic_weights)
            ))

            topic_a = pyro.sample('topic_a', dist.Delta(
                pyro.param('AutoDelta.topic_a', initial_topic_a)
            ).to_event(2))

            topic_b = pyro.sample('topic_b', dist.Delta(
                pyro.param('AutoDelta.topic_b', initial_topic_b)
            ).to_event(2))

    probs = pyro.param('new_participant_topic_q', initial_alpha, constraint=constraints.simplex)
    participant_topics = pyro.sample("new_participant_topic", dist.Delta(probs).to_event(1))
    # transition_topics = pyro.sample("new_transition_topic", dist.Categorical(participant_topics),
    #                                 infer={"enumerate": "parallel"})
    # rowind = obsmat[1].type(torch.long)
    # colind = obsmat[2].type(torch.long)
    # d = dist.Beta(topic_a[transition_topics, rowind, colind],
    #               topic_b[transition_topics, rowind, colind]).to_event(2)
    # pyro.sample('obs', d, obs = obsmat[0])

# new_guide = AutoDelta(poutine.block(model_multi_obs_dim, expose = ['new_participant_topic']))
def initialize_multi_obs_dim(seed, model, guide, data):
    global svi
    pyro.set_rng_seed(seed)
    if 'new_participant_topic_q' in pyro.get_param_store().keys():
        pyro.get_param_store().__delitem__('new_participant_topic_q')
    svi = SVI(model, guide, tm.optim, loss = tm.elbo)
    return svi.loss(model, guide, data)

nsteps = np.arange(.1,1,.1).shape[0]
stor_dim_seed = torch.empty(size = [data.shape[1],data.shape[2],nsteps])
stor_dim_init_loss = torch.empty(size = [data.shape[1],data.shape[2],nsteps])
stor_dim_final_loss = torch.empty(size = [data.shape[1],data.shape[2],nsteps])
stor_dim_prb = torch.empty(size = [data.shape[1],data.shape[2],nsteps, tm.K])

print('here')
# iterate through all features, each with value .1-.9
for row in np.arange(data.shape[1]):
    for col in np.arange(data.shape[2]):
        step_count = 0
        print('({},{})'.format(row,col))
        for step in np.arange(.1, 1,.1):
            newdata = torch.tensor([step, row, col]).float() 
            loss, seed = min((initialize_multi_obs_dim(seed,model_multi_obs_dim,new_guide,newdata),
                              seed) for seed in range(100))
            initialize_multi_obs_dim(seed,model_multi_obs_dim,new_guide,newdata)
            stor_dim_seed[row, col, step_count] = seed
            stor_dim_init_loss[row,col,step_count] = loss
            print(newdata)
            for i in range(2500):
                loss = svi.step(newdata)
                if i % 1000 == 0 :
                    print(loss)
                    print('old', pyro.get_param_store()['AutoDelta.topic_weights'])
                    print('new', pyro.get_param_store()['new_participant_topic_q'])
            stor_dim_final_loss[row,col,step_count] = loss
            stor_dim_prb[row,col,step_count,:] = pyro.get_param_store()['new_participant_topic_q']
            step_count += 1

with open('tomtom_sparse_dim_param.pkl','wb') as f:
    pickle.dump([stor_dim_seed, stor_dim_init_loss, stor_dim_final_loss, stor_dim_prb],f)
