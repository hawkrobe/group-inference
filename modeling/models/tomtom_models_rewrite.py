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
        self.mtype = mtype # 'grp','dim'
        self.target = target # 'self','targ','avg'
        self.dtype = dtype # 'norm','raw'
        self.auto = auto # 'noauto','all'
        self.stickbreak = stickbreak
        # set the parameters inferred through the guide based on the kind of data
        self.exposed_dict = {
            'grp_norm': ['weights', 'concentration'],
            'grp_raw': ['weights', 'alpha', 'beta'],
            'dim_norm': ['topic_weights', 'topic_concentration', 'participant_topics'],
            'dim_raw': ['topic_weights', 'topic_a','topic_b', 'participant_topics']
        }
        self. exposed_params = self.exposed_dict[f'{self.mtype}_{self.dtype}']
        # additional params
        self.data = data
        self.nparticipants = data.shape[0]
        self.nfeatures = data.shape[1]
        self.ncol = data.shape[2]
        # optimizers
        self.optim = pyro.optim.Adam({'lr': 0.0005, 'betas': [0.8, 0.99]})
        self.elbo = TraceEnum_ELBO(max_plate_nesting=1)

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
            weights = pyro.sample('weights', dist.Dirichlet(0.5 * torch.ones(self.K)))
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

    def initialize(self,seed):
        # global global_guide, svi
        pyro.set_rng_seed(seed)
        pyro.clear_param_store()

        self.guide = AutoDelta(poutine.block(self.model, expose = self.exposed_params))
        self.svi = SVI(self.model, self.guide, self.optim, loss = self.elbo)
        # return self.svi.loss(self.model, self.guide, self.data)
        return self.svi.loss(self.model, self.guide) # no longer need to pass data explicitly

    def get_membership(self, temperature):
        guide_trace = poutine.trace(self.guide).get_trace(self.data)  # record the globals
        trained_model = poutine.replay(self.model, trace=guide_trace)  # replay the globals

        inferred_model = infer_discrete(trained_model, temperature=temperature,
                                        first_available_dim=-2)  # avoid conflict with data plate
        trace = poutine.trace(inferred_model).get_trace(self.data)
        return trace.nodes["assignment"]["value"]

    def fit(self, print_fit = True, return_guide = False):
        pyro.clear_param_store()
        #declare dataset to be modeled
        dtname = f't{self.target}_{self.dtype}_{self.auto}_3d'
        if print_fit:
            print("running SVI with: {}".format(dtname))
        # instantiate a model based on self params
        self.model = getattr(self,f'model_{self.mtype}_{self.dtype}')
        # find good starting point
        loss, self.seed = min((self.initialize(seed), seed) for seed in range(100))
        self.initialize(self.seed)
        if print_fit:
            print('seed = {}, initial_loss = {}'.format(self.seed, loss))

        gradient_norms = defaultdict(list)
        for name, value in pyro.get_param_store().named_parameters():
            value.register_hook(lambda g, name=name: gradient_norms[name].append(g.norm().item()))

        self.losses = []
        for i in range(3000):
            loss = self.svi.step() # no longer need to pass data explicitly
            #print(loss)
            self.losses.append(loss)
            if print_fit and i % 100 == 0:
                print('.',end = '')
        if print_fit:
            print('\n final loss: {}\n'.format(self.losses[-1]))

        # code chunk to calculate the likelihood of data once model is fitted
        # modified to take a sample of log prob for each model
        lp_iter = []
        for i in range(500):
            guide_trace = poutine.trace(self.guide).get_trace(self.data)
            model_trace = poutine.trace(poutine.replay(self.model, trace=guide_trace)).get_trace() # no longer need to pass data explicitly
            lp_iter.append(model_trace.log_prob_sum() - guide_trace.log_prob_sum())
        self.logprob_estimate = sum(lp_iter)/len(lp_iter)
        # code chunk to return
        map_estimates = self.guide(self.data)
        if return_guide:
            guidecopy = deepcopy(self.guide)
            if 'grp' in mtype:
                self.membership = self.get_membership(temperature = 0)
                return self.seed, self.map_estimates, self.membership, self.logprob_estimate, self.guidecopy
            elif 'dim' in mtype:
                return self.seed, self.map_estimates, self.logprob_estimate, self.guidecopy
        else:
            if 'gr' in mtype:
                self.membership = self.get_membership(temperature = 0)
                return self.seed, self.map_estimates, self.membership, self.logprob_estimate
            elif 'dim' in mtype:
                return self.seed, self.map_estimates, self.logprob_estimate

if __name__ == '__main__':
    # import pickled data
    import pickle
    with open('../data/tomtom_data_preprocessed.pkl','rb') as f:
        [tself_norm_all_3d, tself_norm_noauto_3d, tself_raw_all_3d, tself_raw_noauto_3d,
        ttarg_norm_all_3d, ttarg_norm_noauto_3d, ttarg_raw_all_3d, ttarg_raw_noauto_3d,
        tavg_norm_all_3d, tavg_norm_noauto_3d, tavg_raw_all_3d, tavg_raw_noauto_3d] = pickle.load(f)

    tomtom = TransitionModel(
        data = tself_raw_noauto_3d,
        K = 3,
        target = 'self',
        dtype = 'raw',
        auto = 'noauto',
        mtype = 'grp'
    )

    tomtom.fit()
