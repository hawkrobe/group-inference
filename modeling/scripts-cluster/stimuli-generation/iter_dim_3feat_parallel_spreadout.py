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

# import pickled data
with open('tomtom_data_preprocessed.pkl','rb') as f:
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

# function to parse command line input on which segment of cellcombo to run
def parse_cellcombo_segment(ind):
    # map segments of cellcombo to index
    endpoint = np.ceil(len(cellcombo)/100)*100
    indmap = np.arange(0, endpoint, 100)
    startpoint = indmap[ind]
    if startpoint == 34200:
        seg = np.arange(startpoint, startpoint + 20)
    else:
        seg = np.arange(startpoint, startpoint + 100)
    return seg.astype(int)

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
comb_segment = int(sys.argv[1])

def combo_3feat(comb):
    comb_counter = cellcombo.index(comb)
    #first identify all row and col indices
    rs = coord.loc[np.array(comb),'row']
    cs = coord.loc[np.array(comb),'col']
    # init storage tensors
    stor_seed = torch.empty(size = [nsteps, nsteps, nsteps])
    stor_init_loss = torch.empty(size = [nsteps, nsteps, nsteps])
    stor_final_loss = torch.empty(size = [nsteps, nsteps, nsteps])
    stor_prb = torch.empty(size = [nsteps, nsteps, nsteps, tm.K])
    for step in stepcombo:
        tik = time.time()
        newdata = torch.tensor(np.stack((step,rs,cs),1)).float()
        step1 = np.where(allsteps == step[0])[0][0] # can try to make this generalize for arbitrary nstim, seems hard tho
        step2 = np.where(allsteps == step[1])[0][0]
        step3 = np.where(allsteps == step[2])[0][0]

        # Find a reasonable initialization over 100 attempts
        loss, seed = min((initialize_multi_obs_dim(seed,model_multi_obs_dim,new_guide,newdata),
                          seed) for seed in range(100))
        initialize_multi_obs_dim(seed,model_multi_obs_dim,new_guide,newdata)
        stor_seed[step1,step2,step3] = seed
        stor_init_loss[step1,step2,step3] = loss
        for i in range(2000):
            loss = svi.step(newdata)
            # if i % 1000 == 0 :
            #     print(loss)
            #     print('old', pyro.get_param_store()['AutoDelta.topic_weights'])
            #     print('new', pyro.get_param_store()['new_participant_topic_q'])
        stor_final_loss[step1,step2,step3] = loss
        stor_prb[step1,step2,step3,:] = pyro.get_param_store()['new_participant_topic_q']
        print(f'step: {step},time: {time.time() - tik}, p_topic: {stor_prb[step1,step2,step3,:]}')

    print('Finished combo {}'.format(comb_counter), flush=True)
    return stor_seed, stor_init_loss, stor_final_loss, stor_prb
    # return (stor_seed, stor_init_loss, stor_final_loss, stor_prb)

# with open('tomtom_sparse_dim_param.pkl','wb') as f:
#     pickle.dump([stor_dim_seed, stor_dim_init_loss, stor_dim_final_loss, stor_dim_prb],f)
# import torch.multiprocessing as multiprocessing
#
# ta = time.time()
# pool = multiprocessing.Pool()
# subcombo = [cellcombo[i] for i in comb_segment]
# # pool_out = pool.map(combo_3feat, subcombo)
# pool_out = []
# for c in subcombo:
#     print(c)
#     pool_out.append(combo_3feat(c))
# stor_dim_seed_3feat = torch.stack([tt[0] for tt in pool_out])
# stor_dim_init_loss_3feat = torch.stack([tt[1] for tt in pool_out])
# stor_dim_final_loss_3feat = torch.stack([tt[2] for tt in pool_out])
# stor_dim_prb_3feat = torch.stack([tt[3] for tt in pool_out])

subcombo = cellcombo[comb_segment]
stor_dim_seed_3feat, stor_dim_init_loss_3feat, stor_dim_final_loss_3feat, stor_dim_prb_3feat = combo_3feat(subcombo)

# store output chunks
fname = 'stor_dim/stor_dim_3feat_chunk_{}.pkl'.format(comb_segment)
with open(fname, 'wb') as f:
    pickle.dump([stor_dim_seed_3feat, stor_dim_init_loss_3feat, stor_dim_final_loss_3feat, stor_dim_prb_3feat], f)
