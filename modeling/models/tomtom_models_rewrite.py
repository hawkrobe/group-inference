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
import torch.nn.functional as F

class TransitionModel():
    '''
    contains all models crossing mtype, dtype, and auto
    automatically determines which model to use based on init params
    '''
    def __init__(self, data, K, target, dtype, auto, mtype, stickbreak = False):
        self.K = 3
        self.mtype = 'group'
        self.target = 'self' # 'self','targ','avg'
        self.dtype = 'raw' # 'norm','raw'
        self.auto = 'noauto' # 'noauto','all'
        self.stickbreak = False

        self.data = data
        self.nparticipants = data.shape[0]
        self.nfeatures = data.shape[1]
        self.ncol = data.shape[2]

    def mix_weights(self, beta):
        beta1m_cumprod = (1 - beta).cumprod(-1)
        return F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)

    @config_enumerate
    def model_grp_norm(self):
        # Background probability of different groups
        if self.stickbreak:
            # stick breaking process for assigning weights to groups
            with pyro.plate("beta_plate", self.K-1):
                beta_mix = pyro.sample("weights", dist.Beta(1, 10))
            weights = self.mix_weights(beta_mix)
        else:
            weights = pyro.sample('weights', dist.Dirichlet(0.5 * torch.ones(K)))
        # model parameters
        with pyro.plate('components', self.K):
            # concentration parameters
            concentration = pyro.sample(
                'concentration',
                dist.Gamma(
                    2 * torch.ones(self.nfeatures,self.ncol),
                    1/3 * torch.ones(self.nfeatures,self.ncol)
                ).to_event(2)
            )
        with pyro.plate('data', self.nparticipants):
            assignment = pyro.sample('assignment', dist.Categorical(weights))
            d = dist.Dirichlet(concentration[assignment,:,:])
            pyro.sample('obs', d.to_event(1), obs=self.data)

    @config_enumerate
    def model_grp_raw(self):
        # Background probability of different groups
        if self.stickbreak:
            # stick breaking process for assigning weights to groups
            with pyro.plate("beta_plate", self.K-1):
                beta_mix = pyro.sample("weights", dist.Beta(1, 10))
            weights = self.mix_weights(beta_mix)
        else:
            weights = pyro.sample('weights', dist.Dirichlet(0.5 * torch.ones(K)))
        # model paramteres
        with pyro.plate('components', self.K):
            alphas = pyro.sample(
                'alpha',
                dist.Gamma(
                    2 * torch.ones(self.nfeatures,self.ncol),
                    1/3 * torch.ones(self.nfeatures,self.ncol)
                ).to_event(2)
            )
            betas = pyro.sample(
                'beta',
                dist.Gamma(
                    2 * torch.ones(self.nfeatures,self.ncol),
                    1/3 * torch.ones(self.nfeatures,self.ncol)
                ).to_event(2)
            )

        with pyro.plate('data', self.nparticipants):
            assignment = pyro.sample('assignment', dist.Categorical(weights))
            d = dist.Beta(alphas[assignment,:,:], betas[assignment,:,:])
            pyro.sample('obs', d.to_event(2), obs=self.data)

    @config_enumerate
    def model_dim_norm(self):
        with pyro.plate('topic', self.K):
            # sample a weight and value for each topic
            topic_weights = pyro.sample("topic_weights", dist.Gamma(1. / self.K, 1.))
            topic_concentration = pyro.sample(
                "topic_concentration",
                dist.Gamma(
                    2 * torch.ones(self.nfeatures,self.ncol),
                    1/3 * torch.ones(self.nfeatures,self.ncol)
                ).to_event(2)
            )

        with pyro.plate('participants', self.nparticipants):
            # sample each participant's idiosyncratic topic mixture
            participant_topics = pyro.sample("participant_topics", dist.Dirichlet(topic_weights))
            transition_topics = pyro.sample("transition_topics",
                dist.Categorical(participant_topics),
                infer={"enumerate": "parallel"}
            )
            # here to_event(1) instead of to_event(2) makes the bastch and event shape line up with the raw data model
            # and makes it run, but make sure it's actually right right (I think it is)
            out = dist.Dirichlet(topic_concentration[transition_topics]).to_event(1)
            data = pyro.sample("obs", out, obs=self.data)

    @config_enumerate
    def model_dim_raw(self):
        with pyro.plate('topic', self.K):
            # sample a weight and value for each topic
            topic_weights = pyro.sample("topic_weights", dist.Gamma(1. / self.K, 1.))
            topic_a = pyro.sample(
                "topic_a",
                dist.Gamma(
                    2 * torch.ones(self.nfeatures,self.ncol),
                    1/3 * torch.ones(self.nfeatures,self.ncol)
                ).to_event(2)
            )
            topic_b = pyro.sample(
                "topic_b",
                dist.Gamma(
                    2 * torch.ones(self.nfeatures,self.ncol),
                    1/3 * torch.ones(self.nfeatures,self.ncol)
                ).to_event(2)
            )
        with pyro.plate('participants', self.nparticipants):
            # sample each participant's idiosyncratic topic mixture
            participant_topics = pyro.sample("participant_topics", dist.Dirichlet(topic_weights))
            transition_topics = pyro.sample(
                "transition_topics",
                dist.Categorical(participant_topics),
                infer={"enumerate": "parallel"}
            )
            out = dist.Beta(topic_a[transition_topics], topic_b[transition_topics]).to_event(2)
            data = pyro.sample("obs", out, obs=self.data)

    def initialize(self):
        pass

    def get_membership(self):
        pass

    def fit(self):
        pass

if __name__ == '__main__':
    pass
