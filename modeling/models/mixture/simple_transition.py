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
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete
from pyro.ops.indexing import Vindex

# change for from collections import defaultdict  demonstration purposes?

pyro.enable_validation(True)

def generate_data(group) :
    big_val = dist.Beta(10, 1)
    small_val = dist.Beta(1, 10)
    # small_val = dist.Normal(0, 0.1)
    # big_val = dist.Normal(5, 0.1)
    if group == 'within' :
        return torch.tensor([small_val.sample(), big_val.sample(), big_val.sample(), small_val.sample()])
    elif group == 'across' :
        return torch.tensor([big_val.sample(), small_val.sample(), small_val.sample(), big_val.sample()])

data = torch.stack([generate_data('across') for i in range(100)] +
                   [generate_data('within') for i in range(100)],
                   dim=0)
print(data)
print(data.shape)

K = 2  # Fixed number of components.

@config_enumerate()
def model(data):
    # Background probability of different groups (assume equally likely)
    weights = torch.tensor([0.5, 0.5])
    with pyro.plate('components', K):
        # concentration parameters
        alphas = pyro.sample('alpha', dist.Gamma(2 * torch.ones(4), 1/3 * torch.ones(4)).to_event(1))
        betas = pyro.sample('beta', dist.Gamma(2 * torch.ones(4), 1/3 * torch.ones(4)).to_event(1))

    with pyro.plate('data', data.shape[0]):
        # Local variables.
        assignment = pyro.sample('assignment', dist.Categorical(weights))
        d = dist.Beta(alphas[assignment], betas[assignment])
        pyro.sample('obs', d.to_event(1), obs=data)

def model_single_obs(obs, i, j):
    # Background probability of different groups (assume equally likely)
    weights = torch.tensor([0.5, 0.5])
    with pyro.plate('components', K):
        # concentration parameters
        alphas = pyro.sample('alpha', dist.Gamma(2 * torch.ones(4), 1/3 * torch.ones(4)).to_event(1))
        betas = pyro.sample('beta', dist.Gamma(2 * torch.ones(4), 1/3 * torch.ones(4)).to_event(1))

    # Local variables.
    assignment = pyro.sample('assignment', dist.Categorical(weights))
    betas = dist.Beta(alphas[assignment], betas[assignment])
    pyro.sample('obs',
                betas[torch.LongTensor(i), torch.LongTensor(j)],
                obs=obs)
       
optim = pyro.optim.Adam({'lr': 0.0005, 'betas': [0.8, 0.99]})
elbo = TraceEnum_ELBO(max_plate_nesting=1)

def initialize(seed):           
    global global_guide, svi
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()
    global_guide = AutoDelta(poutine.block(model, expose=['alpha', 'beta']))
    svi = SVI(model, global_guide, optim, loss=elbo)
    return svi.loss(model, global_guide, data)

loss, seed = min((initialize(seed), seed) for seed in range(100))
initialize(100)
print('seed = {}, initial_loss = {}'.format(seed, loss))

gradient_norms = defaultdict(list)
for name, value in pyro.get_param_store().named_parameters():
    value.register_hook(lambda g, name=name: gradient_norms[name].append(g.norm().item()))

losses = []
for i in range(10000):
    loss = svi.step(data)
    losses.append(loss)
    if i % 100 == 0 :
        print(loss)

map_estimates = global_guide(data)
alphas = map_estimates['alpha']
betas = map_estimates['beta']
print('alphas = {}'.format(alphas.data.numpy()))
print('betas = {}'.format(betas.data.numpy()))

    
