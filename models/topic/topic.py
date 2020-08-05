import os
import sys
from collections import defaultdict
import torch
import numpy as np
import scipy.stats
from torch.distributions import constraints
from matplotlib import pyplot
import random

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.autoguide import AutoDelta
from pyro.optim import Adam
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete
from pyro.ops.indexing import Vindex
from pyro.infer import MCMC, NUTS

## Data preparation and cleaning
# importing required packages
import pyreadr
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

pyro.enable_validation(True)

# import data and convert to python objects
# load Rdata object
rdata = pyreadr.read_r('tomtom_data/det_dfs.Rdata')
# pull out separate dataframes, one with choice text, the other with numeric responses
df = rdata['df']
dfn = rdata['dfn']

# read in state labels
states = pd.read_csv('tomtom_data/states.csv')
states['statepair'] = states['state1'] + '_' + states['state2']# construct state pair strings

# extract all the self ratings
ind_selfq1 = df.columns.get_loc('X1_Q12_1') # location of the first question
trans_self = df.iloc[:,list(range(ind_selfq1,(ind_selfq1+75)))]
print('shape of self-rating dataframe: ',trans_self.shape)
trans_self.columns = states['statepair'].tolist() # renaming transitioin columns with the corresponding transition pairs

# extract all the specific target ratings
ind_targq1 = df.columns.get_loc('X1_Q9_1') # location of the first question
trans_targ = df.iloc[:,list(range(ind_targq1,(ind_targq1+75)))]
print('shape of target-rating dataframe: ', trans_targ.shape)
trans_targ.columns = states['statepair'].tolist() # renaming transitioin columns with the corresponding transition pairs

# extract all the group level ratings
ind_avgq1 = df.columns.get_loc('X1_Q11_1') # location of the first question
trans_avg = df.iloc[:,list(range(ind_avgq1,(ind_avgq1+75)))]
print('shape of group-rating dataframe: ', trans_avg.shape)
trans_avg.columns = states['statepair'].tolist() # renaming transitioin columns with the corresponding transition pairs

# reusable code for pre processing each of the three sets of ratings into pyro compatible format
def data_transform(trans):
    # set the constant for converting probability to frequency
    freq_constant = 10000
    # indexing autotransition columns
    colnames = trans.columns.tolist()
    cnsplit = [p.split('_') for p in colnames]
    idx_autotransition = [p[0] == p[1] for p in cnsplit] # list of boolean, True = is autotransition
    
    # 1. normalizing with autotransitions included, one df for probability, one converted to frequency
    # initialize 2 dataframes
    t_norm_all = pd.DataFrame(columns=trans.columns, index = trans.index)
    t_norm_all_f =  pd.DataFrame(columns=trans.columns, index = trans.index)
    
    # normalize by row-sum every five columns, since the columns are already arranged by from-state in 5
    for i in range(0,trans.shape[1],5):
        dftemp = trans.iloc[:,i:(i+5)]
        dftemp_rowsum = dftemp.sum(axis = 1)
        normed_cols = dftemp/dftemp_rowsum[:,np.newaxis]
        t_norm_all.iloc[:,i:(i+5)] = normed_cols
        t_norm_all_f.iloc[:,i:(i+5)] = (normed_cols * freq_constant).round()
        
    # 2. two additional dataframes: normed with auto transition but don't contain them
    t_norm_all_noauto = t_norm_all.loc[:,[not t for t in idx_autotransition]]
    t_norm_all_noauto_f = t_norm_all_f.loc[:,[not t for t in idx_autotransition]]

    # 3. finally, normalizing without autotransitions, and also convert to frequency
    trans_noauto = trans.loc[:,[not t for t in idx_autotransition]]
    t_norm_noauto = pd.DataFrame(columns=trans_noauto.columns, index = trans_noauto.index)
    t_norm_noauto_f = pd.DataFrame(columns=trans_noauto.columns, index = trans_noauto.index)

    # normalize by row-sum every FOUR columns, grouped by from-state in 4 without autotransition
    for i in range(0,trans_noauto.shape[1],4):
        dftemp = trans_noauto.iloc[:,i:(i+4)]
        dftemp_rowsum = dftemp.sum(axis = 1)
        normed_cols = dftemp/dftemp_rowsum[:,np.newaxis]
        t_norm_noauto.iloc[:,i:(i+5)] = normed_cols
        t_norm_noauto_f.iloc[:,i:(i+5)] = (normed_cols * freq_constant).round()
        
    t_norm_all_3d = torch.tensor(np.stack([np.array(t_norm_all.iloc[0]).reshape(15,5) 
                                        for i in np.arange(t_norm_all.shape[0])]).astype('float32'))
    t_norm_noauto_3d = torch.tensor(np.stack([np.array(t_norm_noauto.iloc[0]).reshape(15,4) 
                                            for i in np.arange(t_norm_noauto.shape[0])]).astype('float32'))
    t_raw_all_3d = torch.tensor(np.stack([np.array(trans.iloc[0])
                                            for i in np.arange(trans.shape[0])]).astype('float32'))
    t_raw_noauto_3d = torch.tensor(np.stack([np.array(trans_noauto.iloc[0])
                                            for i in np.arange(trans_noauto.shape[0])]).astype('float32'))
    
    return t_norm_all_3d, t_norm_noauto_3d, t_raw_all_3d/100, t_raw_noauto_3d/100

global tself_norm_all_3d, tself_norm_noauto_3d, tself_raw_all_3d, tself_raw_noauto_3d
tself_norm_all_3d, tself_norm_noauto_3d, tself_raw_all_3d, tself_raw_noauto_3d = data_transform(trans_self)

print(tself_raw_noauto_3d.shape)

# function used to initialize model
def initialize(seed,model,data):    
    global global_guide, svi
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()
    global_guide = AutoDelta(poutine.block(model, expose=['topic_weights', 'topic_a','topic_b', 'participant_topics']))
    svi = SVI(model, global_guide, optim, loss=elbo)
    return svi.loss(model, global_guide, data)
            
# define a code chunk that does the SVI step for singel variation
def tomtom_svi():
    pyro.clear_param_store()
    
    #declare dataset to be modeled
    dtname = 't{}_{}_{}_3d'.format(target, norm, auto)
    print("running SVI with: {}".format(dtname))
    data = globals()[dtname]
    
    loss, seed = min((initialize(seed,model,data), seed) for seed in range(100))
    initialize(seed,model,data)
    print('seed = {}, initial_loss = {}'.format(seed, loss))
    
    gradient_norms = defaultdict(list)
    for name, value in pyro.get_param_store().named_parameters():
        value.register_hook(lambda g, name=name: gradient_norms[name].append(g.norm().item()))

    losses = []
    for i in range(3000):
        loss = svi.step(data)
        print(loss)
        losses.append(loss)
        if i % 100 == 0:
            print('.',end = '')
    print('\n final loss: {}\n'.format(losses[-1]))

    map_estimates = global_guide(data)
    weights = map_estimates['topic_weights']
    print('weights = {}'.format(weights.data.numpy()))
    if norm == 'norm':
        concentration = map_estimates['concentration']
        print('concentration = {}'.format(concentration.data.numpy()))
    elif norm == 'raw':
        alpha = map_estimates['topic_a']
        print('alphas = {}'.format(alpha.data.numpy()))
        beta = map_estimates['topic_b']
        print('beta = {}'.format(beta.data.numpy()))
    return seed, map_estimates

def tomtom_mcmc(seed,nsample = 5000, burnin = 1000):
    pyro.clear_param_store()
    pyro.set_rng_seed(seed)

    #declare dataset to be modeled
    dtname = 't{}_{}_{}_3d'.format(target, norm, auto)
    print("running MCMC with: {}".format(dtname))
    data = globals()[dtname]

    nuts_kernel = NUTS(model)

    mcmc = MCMC(nuts_kernel, num_samples=nsample, warmup_steps=burnin)
    mcmc.run(data)
    
    posterior_samples = mcmc.get_samples()
    return posterior_samples
                                     
def model(data):
    num_topics = 2
    nparticipants = data.shape[0]
    nfeatures = data.shape[1]

    with pyro.plate('topic', num_topics):
        # sample a weight and value for each topic
        topic_weights = pyro.sample("topic_weights", dist.Gamma(1. / num_topics, 1.))
        topic_a = pyro.sample("topic_a", dist.Gamma(2 * torch.ones(nfeatures),
                                                         1/3 * torch.ones(nfeatures)).to_event(1))
        topic_b= pyro.sample("topic_b", dist.Gamma(2 * torch.ones(nfeatures),
                                                   1/3 * torch.ones(nfeatures)).to_event(1))
        # print('topic weights', topic_weights.shape)
        # print('topic values', topic_a.shape)

    with pyro.plate('participants', nparticipants):
        # sample each participant's idiosyncratic topic mixture
        participant_topics = pyro.sample("participant_topics", dist.Dirichlet(topic_weights))
        #print('participant topics', participant_topics.shape)
        
        transition_topics = pyro.sample("transition_topics", dist.Categorical(participant_topics),
                                        infer={"enumerate": "parallel"})
        #print('topics')
        # print('transition topics', transition_topics)
        # print('indexed', topic_concentrations[transition_topics])
        out = dist.Beta(topic_a[transition_topics], topic_b[transition_topics]).to_event(1)
        # print('observation batch:', out.batch_shape)
        # print('observation event:', out.event_shape)
        # print('data', data.shape)
        data = pyro.sample("obs", out, obs=data)

K = 2
target = 'self' # 'self','targ','avg'
norm = 'raw' # 'norm','raw'
auto = 'noauto' # 'noauto','all'
optim = pyro.optim.Adam({'lr': 0.0005, 'betas': [0.8, 0.99]})
elbo = TraceEnum_ELBO(max_plate_nesting=1)
tomtom_svi()
