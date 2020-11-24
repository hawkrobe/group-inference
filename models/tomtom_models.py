# parameters that need to be passed to the run_model function
# group-like model or dimensional model?
# a priori K vs. stick breaking k
# normed data vs. raw data
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

# define a model function that's dynamically declared

def mix_weights(beta):
    beta1m_cumprod = (1 - beta).cumprod(-1)
    return F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)

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



# function used to initialize model
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

# packing code to read out (predict) cluster membership
# only applicable to group model
def get_membership(model, guide, data, temperature=0):
    guide_trace = poutine.trace(global_guide).get_trace(data)  # record the globals
    trained_model = poutine.replay(model, trace=guide_trace)  # replay the globals

    inferred_model = infer_discrete(trained_model, temperature=temperature,
                                    first_available_dim=-2)  # avoid conflict with data plate
    trace = poutine.trace(inferred_model).get_trace(data)
    return trace.nodes["assignment"]["value"]

# define a code chunk that does the SVI step for singel variation
def tomtom_svi(data, print_fit = True):
    pyro.clear_param_store()

    #declare dataset to be modeled
    dtname = 't{}_{}_{}_3d'.format(target, dtype, auto)
    if print_fit:
        print("running SVI with: {}".format(dtname))
    # data = globals()[dtname]

    loss, seed = min((initialize(seed,model,data), seed) for seed in range(100))
    initialize(seed,model,data)
    if print_fit:
        print('seed = {}, initial_loss = {}'.format(seed, loss))

    gradient_norms = defaultdict(list)
    for name, value in pyro.get_param_store().named_parameters():
        value.register_hook(lambda g, name=name: gradient_norms[name].append(g.norm().item()))

    losses = []
    for i in range(3000):
        loss = svi.step(data)
        #print(loss)
        losses.append(loss)
        if print_fit and i % 100 == 0:
            print('.',end = '')
#             print(loss)
    if print_fit:
        print('\n final loss: {}\n'.format(losses[-1]))

    # code chunk to calculate the likelihood of data once model is fitted
    # modified to take a sample of log prob for each model
    lp_iter = []
    for i in range(500):
        guide_trace = poutine.trace(global_guide).get_trace(data)
        model_trace = poutine.trace(poutine.replay(model, trace=guide_trace)).get_trace(data)
        lp_iter.append(model_trace.log_prob_sum() - guide_trace.log_prob_sum())
    logprob_estimate = sum(lp_iter)/len(lp_iter)
    # code chunk to return
    map_estimates = global_guide(data)
    if 'gr' in mtype:
        membership = get_membership(model, global_guide, data, temperature = 0)
        return seed, map_estimates, membership, logprob_estimate
    elif 'dim' in mtype:
        return seed, map_estimates, logprob_estimate

def print_svi_param(map_estimates):
    for i in map_estimates.keys():
        prm = map_estimates[i]
        print('{} = {}'.format(i, prm.data.numpy()))

def tomtom_mcmc(data,seed,nsample = 5000, burnin = 1000):
    pyro.clear_param_store()
    pyro.set_rng_seed(seed)

    # #declare dataset to be modeled
    # dtname = 't{}_{}_{}_3d'.format(target, dtype, auto)
    # print("running MCMC with: {}".format(dtname))
    # data = globals()[dtname]

    nuts_kernel = NUTS(model)

    mcmc = MCMC(nuts_kernel, num_samples=nsample, warmup_steps=burnin)
    mcmc.run(data)

    posterior_samples = mcmc.get_samples()
    return posterior_samples


def somebullshit():
    print(mtype)
