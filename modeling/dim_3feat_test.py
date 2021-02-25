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

# with open('../data/tomtom_data_preprocessed.pkl','rb') as f:
with open('C:/Users/zhaoz/group-inference/data/tomtom_data_preprocessed.pkl','rb') as f:
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

    # This is a reasonable prior for dirichlet concentrations
    gamma_prior = dist.Gamma(
        2 * torch.ones(nfeatures, ncol),
        1/3 * torch.ones(nfeatures, ncol)
    ).to_event(2)

    with pyro.plate('topic', num_topics):
        # sample a weight and value for each topic
        topic_weights = pyro.sample("topic_weights", dist.Gamma(1. / num_topics, 1.))
        topic_a = pyro.sample("topic_a", gamma_prior)
        topic_b = pyro.sample("topic_b", gamma_prior)

    # sample new participant's idiosyncratic topic mixture
    participant_topics = pyro.sample("new_participant_topic", dist.Dirichlet(topic_weights))

    # we parallelize over the possible topics and pyro automatically weights them by their probs
    transition_topics = pyro.sample("new_transition_topic", dist.Categorical(participant_topics),
                                    infer={"enumerate": "parallel"})

    # expand assignment to make dimensions match
    for r in np.arange(obsmat.shape[0]):
        rowind = obsmat[r,1].type(torch.long)
        colind = obsmat[r,2].type(torch.long)
        print(rowind,colind)
        d = dist.Beta(topic_a[transition_topics, rowind, colind],
                      topic_b[transition_topics, rowind, colind])
        pyro.sample('obs_{}'.format(r), d, obs = obsmat[r,0])

@config_enumerate
def new_guide(obsmat):
    # These are just the previous values we can use to initialize params here
    initial_topic_weights = pyro.get_param_store()['AutoDelta.topic_weights']
    initial_alpha = pyro.get_param_store()['AutoDelta.topic_weights']
    initial_topic_a = pyro.get_param_store()['AutoDelta.topic_a']
    initial_topic_b = pyro.get_param_store()['AutoDelta.topic_b']

    # Use poutine.block to Keep our learned values of global parameters.
    with poutine.block(hide_types=["param"]):

        # This has to match the structure of the model
        with pyro.plate('topic', tm.K):
            # We manually define the AutoDelta params we had from before here
            topic_weights_q = pyro.param('AutoDelta.topic_weights', initial_topic_weights)
            topic_a_q = pyro.param('AutoDelta.topic_a', initial_topic_a)
            topic_b_q = pyro.param('AutoDelta.topic_b', initial_topic_b)

            # Each of the sample statements in the above model needs to have a corresponding
            # statement here where we insert our tuneable params
            pyro.sample("topic_weights", dist.Delta(topic_weights_q))
            pyro.sample('topic_a', dist.Delta(topic_a_q).to_event(2))
            pyro.sample('topic_b', dist.Delta(topic_b_q).to_event(2))

    # We define a new learnable parameter for the new participant that
    # sums to 1 (via constraint) and plug this in as their topic probabilities
    probs = pyro.param('new_participant_topic_q', initial_alpha, constraint=constraints.simplex)
    participant_topics = pyro.sample("new_participant_topic", dist.Delta(probs).to_event(1))

def initialize_multi_obs_dim(seed, model, guide, data):
    global svi
    pyro.set_rng_seed(seed)
    if 'new_participant_topic_q' in pyro.get_param_store().keys():
        pyro.get_param_store().__delitem__('new_participant_topic_q')
    svi = SVI(model, guide, tm.optim, loss = tm.elbo)
    return svi.loss(model, guide, data)


# adapting Robert's code from 1 feature to 3 features
# Three Features
dfr = pd.DataFrame({'row':np.arange(data.shape[1]),'key': 1})
dfc = pd.DataFrame({'col':np.arange(data.shape[2]),'key': 1})
coord =pd.merge(dfr,dfc,on='key').drop('key',1) # mapping each cell to their row/col index, just another way to do cartesian product but indexing later is easier
cellcombo = list(itertools.combinations(np.arange(60),3))
nstim = 3
stepcombo = list(itertools.product(np.arange(.1,1,.1), repeat = nstim))
allsteps = np.arange(.1,1,.1)
nsteps = allsteps.shape[0]
# stor_grp_3feat = torch.empty(size = [len(cellcombo),nsteps,nsteps,nsteps])
# stor_grp_prb_3feat = torch.empty(size = [len(cellcombo),nsteps,nsteps,nsteps,K])
# comb_segment = parse_cellcombo_segment(int(sys.argv[1]))

newdata = torch.tensor(
    [[.1,0,0],
     [.3,1,1],
     [.9,5,2]]
)
newdata2 = torch.tensor(
    [[.9,0,0],
     [.9,1,1],
     [.1,5,2]]
)

# Find a reasonable initialization over 100 attempts
loss, seed = min((initialize_multi_obs_dim(seed,model_multi_obs_dim,new_guide,newdata),
                  seed) for seed in range(100))
initialize_multi_obs_dim(seed,model_multi_obs_dim,new_guide,newdata)

tik = time.time()
for i in range(2000):
    loss = svi.step(newdata)
    if i % 1000 == 0 :
        print(loss)
        print('old', pyro.get_param_store()['AutoDelta.topic_weights'])
        print('new', pyro.get_param_store()['new_participant_topic_q'])

print(time.time() - tik)
print(pyro.get_param_store())


# loss, seed = min((initialize_multi_obs_dim(seed,model_multi_obs_dim,new_guide,newdata2),
#                   seed) for seed in range(100))
# initialize_multi_obs_dim(seed,model_multi_obs_dim,new_guide,newdata2)

# tik = time.time()
# for i in range(2000):
#     loss = svi.step(newdata2)
#     if i % 1000 == 0 :
#         print(loss)
#         print('old', pyro.get_param_store()['AutoDelta.topic_weights'])
#         print('new', pyro.get_param_store()['new_participant_topic_q'])
#
# print(time.time() - tik)
# print(pyro.get_param_store()['new_participant_topic_q'])
