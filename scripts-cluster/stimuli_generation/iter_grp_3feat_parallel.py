##### import packages
#base
import os
import sys
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

# define model
@config_enumerate
def model(data):
    # some parameters can be directly derived from the data passed
    # K = 2
    nparticipants = data.shape[0]
    nfeatures = data.shape[1] # number of rows in each person's matrix
    ncol = data.shape[2]

    if 'gr' in mtype:
        # Background probability of different groups
        if stickbreak:
            # stick breaking process for assigning weights to groups
            with pyro.plate("beta_plate", K-1):
                beta_mix = pyro.sample("weights", dist.Beta(1, 10))
            weights = mix_weights(beta_mix)
        else:
            weights = pyro.sample('weights', dist.Dirichlet(0.5 * torch.ones(K)))

        # declare model parameters based on whether the data are row-normalized
        if dtype == 'norm':
            with pyro.plate('components', K):
                # concentration parameters
                concentration = pyro.sample('concentration',
                                            dist.Gamma(2 * torch.ones(nfeatures,ncol), 1/3 * torch.ones(nfeatures,ncol)).to_event(2))

            with pyro.plate('data', data.shape[0]):
                assignment = pyro.sample('assignment', dist.Categorical(weights))
                #d = dist.Dirichlet(concentration[assignment,:,:].clone().detach()) # .detach() might interfere with backprop
                d = dist.Dirichlet(concentration[assignment,:,:])
                pyro.sample('obs', d.to_event(1), obs=data)

        elif dtype == 'raw':
            with pyro.plate('components', K):
                alphas = pyro.sample('alpha', dist.Gamma(2 * torch.ones(nfeatures,ncol), 1/3 * torch.ones(nfeatures,ncol)).to_event(2))
                betas = pyro.sample('beta', dist.Gamma(2 * torch.ones(nfeatures,ncol), 1/3 * torch.ones(nfeatures,ncol)).to_event(2))

            with pyro.plate('data', data.shape[0]):
                assignment = pyro.sample('assignment', dist.Categorical(weights))
                d = dist.Beta(alphas[assignment,:,:], betas[assignment,:,:])
                pyro.sample('obs', d.to_event(2), obs=data)

    elif 'dim' in mtype:
        # stickbreaking still has to be implemented
        # if stickbreak:
        #
        # else:

        # declare model parameters based on whether the data are row-normalized
        if dtype == 'norm':
            with pyro.plate('topic', K):
                # sample a weight and value for each topic
                topic_weights = pyro.sample("topic_weights", dist.Gamma(1. / K, 1.))
                topic_concentration = pyro.sample("topic_concentration", dist.Gamma(2 * torch.ones(nfeatures,ncol),
                                                                 1/3 * torch.ones(nfeatures,ncol)).to_event(2))

            with pyro.plate('participants', nparticipants):
                # sample each participant's idiosyncratic topic mixture
                participant_topics = pyro.sample("participant_topics", dist.Dirichlet(topic_weights))
                transition_topics = pyro.sample("transition_topics", dist.Categorical(participant_topics),
                                                infer={"enumerate": "parallel"})
                # here to_event(1) instead of to_event(2) makes the bastch and event shape line up with the raw data model
                # and makes it run, but make sure it's actually right right (I think it is)
                out = dist.Dirichlet(topic_concentration[transition_topics]).to_event(1)
                data = pyro.sample("obs", out, obs=data)

        elif dtype == 'raw':
            with pyro.plate('topic', K):
                # sample a weight and value for each topic
                topic_weights = pyro.sample("topic_weights", dist.Gamma(1. / K, 1.))
                topic_a = pyro.sample("topic_a", dist.Gamma(2 * torch.ones(nfeatures,ncol),
                                                                 1/3 * torch.ones(nfeatures,ncol)).to_event(2))
                topic_b = pyro.sample("topic_b", dist.Gamma(2 * torch.ones(nfeatures,ncol),
                                                           1/3 * torch.ones(nfeatures,ncol)).to_event(2))

            with pyro.plate('participants', nparticipants):
                # sample each participant's idiosyncratic topic mixture
                participant_topics = pyro.sample("participant_topics", dist.Dirichlet(topic_weights))
                transition_topics = pyro.sample("transition_topics", dist.Categorical(participant_topics),
                                                infer={"enumerate": "parallel"})
                out = dist.Beta(topic_a[transition_topics], topic_b[transition_topics]).to_event(2)
                data = pyro.sample("obs", out, obs=data)

def initialize(seed,model,data):
    global global_guide, svi
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()
    exposed_params = []
    # set the parameters inferred through the guide based on the kind of data
    if 'gr' in mtype:
        if dtype == 'norm':
            exposed_params = ['weights', 'concentration']
        elif dtype == 'raw':
            exposed_params = ['weights', 'alpha', 'beta']
    elif 'dim' in mtype:
        if dtype == 'norm':
            exposed_params = ['topic_weights', 'topic_concentration', 'participant_topics']
        elif dtype == 'raw':
            exposed_params = ['topic_weights', 'topic_a','topic_b', 'participant_topics']

    global_guide = AutoDelta(poutine.block(model, expose = exposed_params))
    svi = SVI(model, global_guide, optim, loss = elbo)
    return svi.loss(model, global_guide, data)

# set model fitting params
K = 3
mtype = 'group'
target = 'self' # 'self','targ','avg'
dtype = 'raw' # 'norm','raw'
auto = 'noauto' # 'noauto','all'
stickbreak = False
optim = pyro.optim.Adam({'lr': 0.0005, 'betas': [0.8, 0.99]})
elbo = TraceEnum_ELBO(max_plate_nesting=1)

# model fitting
pyro.clear_param_store()
#declare dataset to be modeled
dtname = 't{}_{}_{}_3d'.format(target, dtype, auto)
print("running SVI with: {}".format(dtname))
# data = globals()[dtname]
data = vars()[dtname]

loss, seed = min((initialize(seed,model,data), seed) for seed in range(100))
initialize(seed,model,data)
print('seed = {}, initial_loss = {}'.format(seed, loss))

gradient_norms = defaultdict(list)
for name, value in pyro.get_param_store().named_parameters():
    value.register_hook(lambda g, name=name: gradient_norms[name].append(g.norm().item()))

losses = []
for i in range(3000):
    loss = svi.step(data)
    #print(loss)
    losses.append(loss)
    if i % 100 == 0:
        print('.',end = '')
#             print(loss)
print('\n final loss: {}\n'.format(losses[-1]))

# recording output from group model fitting for later use
seed_group = seed
map_group = global_guide(data)
guide_group = global_guide

# defining sparse-input model
@config_enumerate
def model_multi_obs_grp(obsmat):
    # some parameters can be directly derived from the data passed
    # K = 2
    nparticipants = data.shape[0]
    nfeatures = data.shape[1] # number of rows in each person's matrix
    ncol = data.shape[2]

    # Background probability of different groups
    if stickbreak:
        # stick breaking process for assigning weights to groups
        with pyro.plate("beta_plate", K-1):
            beta_mix = pyro.sample("weights", dist.Beta(1, 10))
        weights = mix_weights(beta_mix)
    else:
        weights = pyro.sample('weights', dist.Dirichlet(0.5 * torch.ones(K)))
    # declare model parameters based on whether the data are row-normalized
    if dtype == 'norm':
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

    elif dtype == 'raw':
        with pyro.plate('components', K):
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
guide_trace = poutine.trace(guide_group).get_trace(data)  # record the globals
trained_model_multi = poutine.replay(model_multi_obs_grp, trace = guide_trace)

def classifier_multi_obs(obsmat, temperature): # temperature = 1 to sample
    inferred_model = infer_discrete(trained_model_multi, temperature=temperature,
                                    first_available_dim=-1)  # avoid conflict with data plate
    trace = poutine.trace(inferred_model).get_trace(obsmat.float())
    return trace.nodes["assignment"]["value"]

# # Single Feature
# # initialize storage
niter = 200
# nsteps = np.arange(.1,1,.1).shape[0]
# stor_grp = torch.empty(size = [data.shape[1],data.shape[2],nsteps])
# stor_grp_prb = torch.empty(size = [data.shape[1],data.shape[2],nsteps,K])
# # iterate through all features, each with value .1-.9
# for row in np.arange(data.shape[1]):
#     for col in np.arange(data.shape[2]):
#         step_count = 0
#         for step in np.arange(.1,1,.1):
#             newdata = torch.tensor([step, row, col]).unsqueeze(0)
#             # first MAP classification
#             grp = classifier_multi_obs(newdata, temperature = 0)
#             stor_grp[row,col,step_count] = grp
#             # second use sampling to get group prob
#             stor = torch.zeros(niter)
#             for it in np.arange(niter):
#                 stor[it] = classifier_multi_obs(newdata, temperature = 1)
#             grp_prb = [(stor == i).sum()/float(len(stor)) for i in np.arange(K)]
#             stor_grp_prb[row,col,step_count,:] = torch.tensor(grp_prb)
#             step_count += 1

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

# tout = time.time()
# comb_counter = 0
# for comb in cellcombo[0:100]:
#     t = time.time()
#     #first identify all row and col indices
#     rs = coord.loc[np.array(comb),'row']
#     cs = coord.loc[np.array(comb),'col']
#     for step in stepcombo:
#         newdata = torch.tensor(np.stack((step,rs,cs),1))
#         step1 = np.where(allsteps == step[0])[0][0] # can try to make this generalize for arbitrary nstim, seems hard tho
#         step2 = np.where(allsteps == step[1])[0][0]
#         step3 = np.where(allsteps == step[2])[0][0]
#         # first MAP classification
#         grp = classifier_multi_obs(newdata, temperature = 0)
#         # storage
#         stor_grp_3feat[comb_counter,step1,step2,step3] = grp
# #         # second use sampling to get group prob
# #         stor = torch.zeros(niter)
# #         for it in np.arange(niter):
# #             stor[it] = classifier_multi_obs(newdata, temperature = 1)
# #         grp_prb = [(stor == i).sum()/float(len(stor)) for i in np.arange(K)]
# #         # storage
# #         stor_grp_prb_3feat[comb_counter,step1,step2,step3,:] = torch.tensor(grp_prb)
#     print('comleted combo no. ', comb_counter, '; time: ', time.time()-t)
#     comb_counter += 1
# t100 = time.time() - tout
# print(t100)
# stor_grp_3feat_nopara = stor_grp_3feat

def combo_3feat(comb):
    comb_counter = cellcombo.index(comb)
    #first identify all row and col indices
    rs = coord.loc[np.array(comb),'row']
    cs = coord.loc[np.array(comb),'col']
    # print(comb_counter)
    stor = torch.empty(size = [nsteps, nsteps, nsteps])
    stor_prb = torch.empty(size = [nsteps, nsteps, nsteps, K])
    for step in stepcombo:
        newdata = torch.tensor(np.stack((step,rs,cs),1))
        step1 = np.where(allsteps == step[0])[0][0] # can try to make this generalize for arbitrary nstim, seems hard tho
        step2 = np.where(allsteps == step[1])[0][0]
        step3 = np.where(allsteps == step[2])[0][0]
        # first MAP classification
        grp = classifier_multi_obs(newdata, temperature = 0)
        stor[step1,step2,step3] = grp
        #second use sampling to get group prob
        s = torch.zeros(niter)
        for it in np.arange(niter):
            s[it] = classifier_multi_obs(newdata, temperature = 1)
        grp_prb = [(s == i).sum()/float(len(s)) for i in np.arange(K)]
        stor_prb[step1,step2,step3,:] = torch.tensor(grp_prb)
    print('Finished combo {}'.format(comb_counter))
    return stor, stor_prb

import multiprocessing

ta = time.time()
pool = multiprocessing.Pool()
pool_out = pool.map(combo_3feat, cellcombo[0:20])
stor_grp_3feat = torch.stack([tt[0] for tt in pool_out])
stor_grp_prb_3feat = torch.stack([tt[1] for tt in pool_out])
print(time.time() - ta)
print(stor_grp_3feat.shape)
print(stor_grp_prb_3feat.shape)
print(stor_grp_prb_3feat[5])
