# set up module context
import os
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
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete, Predictive
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
import torch.nn.functional as F
from itertools import combinations

# compute and return a list of K tensors, each tensor the means of a cluster
def get_cluster_summary():
    data = globals()['t{}_{}_{}_3d'.format(target, norm, auto)]
    cluster_means = [data[torch.where(mem_self_norm_all == i)[0]].mean(axis = 0) for i in np.arange(K)]
    return cluster_means

# specific function to get by-cluster density facet grid
def draw_density_group(data, mmap, mmem):
#     data = globals()['t{}_{}_{}_3d'.format(target, norm, auto)]
#     mmap = vars()['map_{}_{}_{}'.format(target, norm, auto)]
#     mmem = vars()['mem_{}_{}_{}'.format(target, norm, auto)]
    filter_crit = .1
    clust_filter = torch.where(mix_weights(mmap['weights']) >= filter_crit)[0] # filter leaving only clusters with mixing weight >.1
    print('{} clusters satisfy the filter requirement of having a mixture weight >= {}\n'.format(clust_filter.shape, filter_crit))

    mem_f = mmem[np.where(np.isin(mmem, clust_filter))[0]]
    data_f = data[np.where(np.isin(mmem, clust_filter))[0]]
    print('these clusters contain {} out of the {} total observations'.format(data_f.shape[0],data.shape[0]))

    # getting the filtered data into long format with group identifier
    df = pd.DataFrame(columns = ['pair','variable','value','cluster'])
    for i in clust_filter.tolist():
        d_clust = data[torch.where(mmem == i)[0]]
        print('cluster {} contains {} observations'.format(i, d_clust.shape[0]))
        nd_clust = np.array(d_clust)
        nd_clust = nd_clust.reshape(nd_clust.shape[0],-1).transpose() #3d to 2d,n_pair(row) X n_mem(column)
        df_clust = pd.DataFrame(data = nd_clust)
        df_clust['pair'] = np.arange(nd_clust.shape[0])
        df_clust = df_clust.melt(id_vars = ['pair'])
        df_clust['cluster'] = i
        df = df.append(df_clust)

    # draw the thing
    g = sns.FacetGrid(df, col="pair",hue = 'cluster', col_wrap=data.shape[2], height=2, xlim = [0,1])
    g.map(sns.kdeplot,'value')

def draw_density_better_group(data, mmap, mmem):
    # draw 3 5*5s, including state labels
#    data = globals()['t{}_{}_{}_3d'.format(target, norm, auto)]
#     mmap = vars()['map_{}_{}_{}'.format(target, norm, auto)]
#     mmem = vars()['mem_{}_{}_{}'.format(target, norm, auto)]
    filter_crit = .1
    clust_filter = torch.where(mix_weights(mmap['weights']) >= filter_crit)[0] # filter leaving only clusters with mixing weight >.1
    print('{} clusters satisfy the filter requirement of having a mixture weight >= {}\n'.format(clust_filter.shape, filter_crit))

    mem_f = mmem[np.where(np.isin(mmem, clust_filter))[0]]
    data_f = data[np.where(np.isin(mmem, clust_filter))[0]]
    print('these clusters contain {} out of the {} total observations'.format(data_f.shape[0],data.shape[0]))

    # getting the filtered data into long format with group identifier
    df = pd.DataFrame(columns = ['pair','from','to','set','variable','value','cluster'])
    # state label array changes based on data being fitted
    for i in clust_filter.tolist():
        d_clust = data[torch.where(mmem == i)[0]]
        print('cluster {} contains {} observations'.format(i, d_clust.shape[0]))
        nd_clust = np.array(d_clust)
        nd_clust = nd_clust.reshape(nd_clust.shape[0],-1).transpose() #3d to 2d,n_pair(row) X n_mem(column)
        # this step differs based on whether autotransitions are included
        if data.shape[2] == 5:
            df_clust = pd.DataFrame(data = nd_clust)
        elif data.shape[2] == 4:
            df_clust = pd.DataFrame(np.zeros((states.shape[0],nd_clust.shape[1])))
#             df_clust[:] = np.nan
            df_clust.loc[states['state1'] != states['state2'],:] = nd_clust
        df_clust['pair'] = np.arange(states.shape[0])
        df_clust['from'] = states['state1']
        df_clust['to'] = states['state2']
        df_clust['set'] = states['set']
        df_clust = df_clust.melt(id_vars = ['pair','from','to','set'])
        df_clust['cluster'] = i
#         print(df_clust)
        # add state labels
#         df_clust['from'] =  np.tile(states['state1'], d_clust.shape[0])
#         df_clust['to'] =  np.tile(states['state2'], d_clust.shape[0])
#         df_clust['set'] =  np.tile(states['set'], d_clust.shape[0])
        df = df.append(df_clust)

    # split into three df one for each set of states
    df_s1 = df.loc[df['set'] == 1]
    df_s1 = df_s1.sort_values(['from','to'])
#     print(df_s1)
    df_s2 = df.loc[df['set'] == 2]
    df_s2 = df_s2.sort_values(['from','to'])
    df_s3 = df.loc[df['set'] == 3]
    df_s3 = df_s3.sort_values(['from','to'])

    # draw the thing
#     g = sns.FacetGrid(df_s1, row = 'from', col="pair",hue = 'cluster', height=2, xlim = [0,1])
    g1 = sns.FacetGrid(df_s1, row = 'from', col="to",hue = 'cluster',xlim = [0,1])
    g1.map(sns.kdeplot,'value')
    g2 = sns.FacetGrid(df_s2, row = 'from', col="to",hue = 'cluster',xlim = [0,1])
    g2.map(sns.kdeplot,'value')
    g3 = sns.FacetGrid(df_s3, row = 'from', col="to",hue = 'cluster',xlim = [0,1])
    g3.map(sns.kdeplot,'value')

# function that generates random new data of size niter for each of the possible clusters, given MAP estimate
# return a four dimensional tensor - d1: cluster, d2: iter, d3&d4: data dimensions
# only for group model
def generate_given_map_grp(dtype, mmap, niter = 1000):
    if dtype == 'all':
        templist = [torch.stack([dist.Dirichlet(mmap['concentration'][i,:,:]).sample() for j in range(niter)])
                for i
                in range(mmap['weights'].shape[0] + 1)]
    elif dtype == 'raw':
        templist = [torch.stack([dist.Beta(mmap['alpha'][i,:,:], mmap['beta'][i,:,:]).sample() for j in range(niter)])
                for i
                in range(mmap['weights'].shape[0] + 1)]
    return torch.stack(templist)

# auxiliary function that turns a 3d tensor into a dataframe with specific desired dimensionalities
def to_dataframe(d_clust):
    nd_clust = np.array(d_clust)
    nd_clust = nd_clust.reshape(nd_clust.shape[0],-1).transpose() #3d to 2d,n_pair(row) X n_mem(column)
    # this step differs based on whether autotransitions are included
    if d_clust.shape[2] == 5:
        df_clust = pd.DataFrame(data = nd_clust)
    elif d_clust.shape[2] == 4:
        df_clust = pd.DataFrame(np.zeros((states.shape[0],nd_clust.shape[1])))
#             df_clust[:] = np.nan
        df_clust.loc[states['state1'] != states['state2'],:] = nd_clust
    df_clust['pair'] = np.arange(states.shape[0])
    df_clust['from'] = states['state1']
    df_clust['to'] = states['state2']
    df_clust['set'] = states['set']
    df_clust = df_clust.melt(id_vars = ['pair','from','to','set'])
    return df_clust

# function that generates a facit grid of density plots for a given cluster,
# plotting real data distribution against distribution of data generated from MAP
# REQUIRES passing pre-generated a chosen cluster
# only for group model
def plot_real_against_generated_grp(data, mmap, mmem, gendata, clust):
    if clust >= (mmap['weights'].shape[0] + 1): # weights have 9 elements, because it's mixed into 10, hence < +1 required
        print('better choose a different cluster')
        return -999
#    data = globals()['t{}_{}_{}_3d'.format(target, norm, auto)]
    d_clust = data[torch.where(mmem == clust)[0]]
    gd_clust = gendata[clust,:,:,:]
    # convert the data and the generated data each into a dataframe
    df = to_dataframe(d_clust)
    gdf = to_dataframe(gd_clust)
    df['type'] = 'empirical'
    gdf['type'] = 'generated'

    # split into three df one for each set of states
    df_s1 = df.loc[df['set'] == 1].append(gdf.loc[gdf['set'] == 1])
    df_s1 = df_s1.sort_values(['from','to'])
    df_s2 = df.loc[df['set'] == 2].append(gdf.loc[gdf['set'] == 2])
    df_s2 = df_s2.sort_values(['from','to'])
    df_s3 = df.loc[df['set'] == 3].append(gdf.loc[gdf['set'] == 3])
    df_s3 = df_s3.sort_values(['from','to'])

    g1 = sns.FacetGrid(df_s1, row = 'from', col="to",hue = 'type',xlim = [0,1],legend_out = True)
    g1.map(sns.kdeplot,'value').add_legend()
    g2 = sns.FacetGrid(df_s2, row = 'from', col="to",hue = 'type',xlim = [0,1],legend_out = True)
    g2.map(sns.kdeplot,'value').add_legend()
    g3 = sns.FacetGrid(df_s3, row = 'from', col="to",hue = 'type',xlim = [0,1],legend_out = True)
    g3.map(sns.kdeplot,'value').add_legend()

# function that calculates the discriminant ability for each of the transition pairs
# given n passing clusters, calculate pair wise ks values
# return a dataframe, with each of the pair wise values, as well as means across all pairs for each transition
def transition_pairwise_ks(data, mmap, mmem):
#    data = globals()['t{}_{}_{}_3d'.format(target, norm, auto)]
    filter_crit = .1
    clust_filter = torch.where(mix_weights(mmap['weights']) >= filter_crit)[0] # filter leaving only clusters with mixing weight >.1
    # keep only clusters (filtered) that have more than 1 data point
    # scipy's ks test function does not throw exception when one of the samples has only one datapoint
    clust_filter = clust_filter[[data[torch.where(mmem == i)[0]].shape[0] > 1 for i in clust_filter]]
    clust_pairs = list(combinations(clust_filter, 2))
    # initialize empty matrix for the ks test results
    ks_stats = np.zeros([data.shape[1] * data.shape[2], len(clust_pairs)])

    for j in range(len(clust_pairs)):
        d1 = data[torch.where(mmem == clust_pairs[j][0])[0]]
        d2 = data[torch.where(mmem == clust_pairs[j][1])[0]]
        # reshape to n_sub * n_statepair 2d matrix
        d1 = d1.reshape(d1.shape[0],-1)
        d2 = d2.reshape(d2.shape[0],-1)
        for i in range(d1.shape[1]):
            ks_stats[i,j] = scipy.stats.ks_2samp(d1[:,i],d2[:,i]).statistic

    # convert to dataframe
    if data.shape[2] == 5:
        df_ks = pd.DataFrame(data = ks_stats)
    elif data.shape[2] == 4:
        df_ks = pd.DataFrame(np.zeros((states.shape[0],ks_stats.shape[1])))
        df_ks[:] = np.nan
        df_ks.loc[states['state1'] != states['state2'],:] = ks_stats
    df_ks.columns = clust_pairs
    df_ks['mean_ks'] = df_ks.mean(axis = 1)
    df_ks['pair'] = np.arange(states.shape[0])
    df_ks['from'] = states['state1']
    df_ks['to'] = states['state2']
    df_ks['set'] = states['set']

    return df_ks

# function that calculates the mean log likelihood for a cluster's data points given the estimated distribution
def mean_cluster_log_prob(mmap, mmem):
    data = globals()['t{}_{}_{}_3d'.format(target, norm, auto)]
    print(data.shape[2])
    print(K)
    # do things differently for normed and raw data
    if norm == 'norm':
        # initialize storage arrays
        n_inclust = np.zeros(K)
        mean_log_prob = np.zeros(K,15)
        for i in range(K):
            n_inclust[i] = (mmem == i).sum()
            if n_inclust == 0:
                mean_log_prob[i] = np.zeros(15) * np.nan
            else:
                c_data = dat[mmem == i]
                c_param = mmap['concentration'][i]
                mean_log_prob[i] = dist.Dirichlet(c_param).log_prob(c_data).mean(axis = 0)
    elif norm == 'raw':
        # initialize storage arrays
        n_inclust = np.zeros(K)
        mean_log_prob = np.zeros([K,15,data.shape[2]])
        for i in range(K):
            n_inclust[i] = (mmem == i).sum()
            if n_inclust[i] == 0:
                mean_log_prob[i] = np.zeros([15,data.shape[2]]) * np.nan
            else:
                c_data = data[mmem == i]
                c_param_a = mmap['alpha'][i]
                c_param_b = mmap['beta'][i]
                print(c_data.shape)
                mean_log_prob[i] = dist.Beta(c_param_a,c_param_b).log_prob(c_data).mean(axis = 0).detach().numpy()

    return mean_log_prob

# function to calculate overall log likelihood of data for the dimensional model after fitting
def data_log_prob_dim(data, mmap, dtype):
    """
    DEFUNCT
    This function calculates the total log probability of the data
    The likelihood of each participant's data under each topic is calculated
    The likehoods are then weighed by participant level topic weights and summed
    Finally, sum over all participants to get the total log probability
    """
    nparticipants = data.shape[0]
    ntopic = mmap['topic_weights'].shape[0]
    p_tpc = mmap['participant_topics']
    if dtype == 'norm':
        tpc = mmap['topic_concentration']
    elif dtype == 'raw':
        ta = mmap['topic_a']
        tb = mmap['topic_b']
    total_log_prob = [] # make sure this is valid for probability DENSITY
    for i in range(nparticipants):
        participant_prob = []
        for j in range(ntopic):
            # get log_prob for participant i's data under topic j
            if dtype == 'norm':
                topic_log_prob = dist.Dirichlet(tpc[j]).log_prob(data[i]).detach().numpy()
            elif dtype == 'raw':
                topic_log_prob = dist.Beta(ta[j],tb[j]).log_prob(data[i]).detach().numpy()
            # exponentiate so we can marginalize over topics
            topic_prob_weighted = np.exp(topic_log_prob) * p_tpc[i,j].detach().numpy()
            participant_prob.append(topic_prob_weighted)
        sum_p_prob = sum(participant_prob) # since all the prob density are from different dists, IS IT PROPER TO SUM THEM UP? if not what to do
        total_log_prob.append(np.log(sum_p_prob))
    sum_total_log_prob = sum(sum(total_log_prob))
    return sum_total_log_prob
