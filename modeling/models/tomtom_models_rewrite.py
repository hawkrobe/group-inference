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
import time

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.autoguide import AutoDelta
from pyro.optim import Adam
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete, Predictive
from pyro.ops.indexing import Vindex
from pyro.infer import MCMC, NUTS
import torch.nn.functional as F
import multiprocessing
import concurrent
import queue

from sklearn.model_selection import KFold



class TransitionModel():
    '''
    contains all models crossing mtype, dtype, and auto
    automatically determines which model to use based on init params
    '''
    def __init__(self, data, K, target, dtype, auto, mtype, stickbreak = False, sparse_nstep = 2000):
        self.K = K
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

        self.sparse_nstep = sparse_nstep

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
            weights = pyro.sample('weights', dist.Dirichlet(0.5 * torch.ones(self.K)))
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
        trace = poutine.trace(inferred_model).get_trace() # no longer passing data explicitly
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
        self.map_estimates = self.guide(self.data)
        if return_guide:
            guidecopy = deepcopy(self.guide)
            if 'gr' in self.mtype:
                self.membership = self.get_membership(temperature = 0)
                return self.seed, self.map_estimates, self.membership, self.logprob_estimate, self.guidecopy
            elif 'dim' in self.mtype:
                return self.seed, self.map_estimates, self.logprob_estimate, self.guidecopy
        else:
            if 'gr' in self.mtype:
                self.membership = self.get_membership(temperature = 0)
                return self.seed, self.map_estimates, self.membership, self.logprob_estimate
            elif 'dim' in self.mtype:
                return self.seed, self.map_estimates, self.logprob_estimate

    def fit_mcmc(self, nsample = 5000, burnin = 1000, seed = 0):
        '''
        to be improved
        '''
        pyro.clear_param_store()
        if hasattr(self, 'seed'):
            pyro.set_rng_seed(self.seed)
        else:
            pyro.set_rng_seed(seed)
        nuts_kernel = NUTS(self.model)
        mcmc = MCMC(nuts_kernel, num_samples=nsample, warmup_steps=burnin)
        mcmc.run(self.data)

        posterior_samples = mcmc.get_samples()
        return posterior_samples

class StimuliSelector():
    '''
    1. given sparse data and fitted params (currently from k=3 model),
        classify(hard/soft assignmet to group)/ infer dimension values (dimension)
    2. sparse data consists of nfeature (cuurently 1 or 3), all possible values on each feature
    3. (group) once generated, use the sparse classification to construct distributions/expected values for unobserved features
    4. (dimension) tbd
    '''
    def __init__(self, fitted = None):
        # option to construct StimuliSelector from fitted TransitionModel objects
        if fitted:
            self = fitted
            self.constructed_from_fitted = True
        else:
            self.constructed_from_fitted = False

    def hard_grp_raw_infer_unobserved(self, stor, map_est = None):
        '''
        Under hard assignment, there are only K possible predictions for each transition
        return K matrices of expected values and (optionally) K matrices of distribution objs
        '''
        # handle when MAP is not available
        if not self.constructed_from_fitted and not map_est:
            raise AttributeError('StimuliSelector must be constructed from a TransitionModel, or MAP estimates must be passed')
        elif self.constructed_from_fitted:
            map_est = self.map_estimates
        # under hard assignment, getting dist and mean is simply indexing
        # construct iterator to loop through
        it_stor =  np.ndenumerate(stor)
        # empty array to store the mixture objects
        dists = np.empty(stor.shape, dtype=object)
        means = np.empty(stor.shape, dtype=object)
        # extract alpha & beta from MAP, should already have the right dimensionality (K*15*4)
        all_a = map_est['alpha']
        all_b = map_est['beta']
        all_dist = []
        for i in range(all_a.shape[0]):
            all_dist.append(dist.Beta(all_a[i],all_b[i]))
        all_mean = [d.mean for d in all_dist]
        for fc in it_stor:
            ind = fc[0]
            # print(ind)
            grp = int(fc[1])
            dists[ind] = all_dist[grp]
            means[ind] = all_mean[grp]
        return dists, means

    def soft_grp_raw_infer_unobserved(self, stor, map_est = None):
        '''
        Under soft assignment
        1. for each feature-value combo extract the assgn proba
        2. use proba to make mixture model
        3. each feature-value combo will have a matrix of expected value and (opt) a matrix of distributions
        might have memory problems
        stor: stored [feature (set) - feature value - assignment proba], must be nd dimensions where nd = 1 + nfeature +1
        dim 0 indexes feature sets, each of the nfeature dims encode values within a feature, and last dimension is K assignment probas
        '''
        # handle when MAP is not available
        if not self.constructed_from_fitted and not map_est:
            raise AttributeError('StimuliSelector must be constructed from a TransitionModel, or MAP estimates must be passed')
        elif self.constructed_from_fitted:
            map_est = self.map_estimates
        # body
        # construct iterator to loop through the first nd-1 dimensions
        last_dim = len(stor.shape)-1
        stor_short = stor.sum(axis = last_dim) # shaving off the last dimension
        it_stor =  np.ndenumerate(stor_short)
        # empty array to store the mixture objects
        mix_dists = np.empty(stor_short.shape, dtype=object)
        mix_means = np.empty(stor_short.shape, dtype=object)
        # extract alpha & beta from MAP, should already have the right dimensionality (K*15*4)
        all_a = map_est['alpha']
        all_b = map_est['beta']
        # construct the component distributions (betas)
        all_dist = dist.Beta(all_a, all_b).to_event(len(all_a.shape)-1)
        # loop through and generate distribution objects
        for fc in it_stor:
            inds = fc[0]
            # print(inds)
            ass_proba = dist.Categorical(stor[inds])
            mix = torch.distributions.mixture_same_family.MixtureSameFamily(ass_proba,all_dist)
            mix_dists[inds] = mix
            mix_means[inds] = mix.mean
        return mix_dists, mix_means

    def dim_raw_infer_unobserved(self, stor, map_est = None):
        # handle when MAP is not available
        if not self.constructed_from_fitted and not map_est:
            raise AttributeError('StimuliSelector must be constructed from a TransitionModel, or MAP estimates must be passed')
        elif self.constructed_from_fitted:
            map_est = self.map_estimates
        # basically the same code as soft_grp, think about how to remove redundancy
        # construct iterator to loop through the first nd-1 dimensions
        last_dim = len(stor.shape)-1
        stor_short = stor.sum(axis = last_dim) # shaving off the last dimension
        it_stor =  np.ndenumerate(stor_short.detach().numpy())
        # empty array to store the mixture objects
        mix_dists = np.empty(stor_short.shape, dtype=object)
        mix_means = np.empty(stor_short.shape, dtype=object)
        # extract relevant MAP map_estimates
        # extract alpha & beta from MAP, should already have the right dimensionality (K*15*4)
        all_a = map_est['topic_a']
        all_b = map_est['topic_b']
        # construct the component distributions (betas)
        all_dist = dist.Beta(all_a, all_b).to_event(len(all_a.shape)-1)
        # loop through and generate distribution objects
        for fc in it_stor:
            inds = fc[0]
            # print(inds)
            ass_proba = dist.Categorical(stor[inds])
            mix = torch.distributions.mixture_same_family.MixtureSameFamily(ass_proba,all_dist)
            mix_dists[inds] = mix
            mix_means[inds] = mix.mean
        return mix_dists, mix_means


class SparseModel(TransitionModel):
    # currently only implemented for raw data models
    @config_enumerate
    def model_multi_obs_grp_raw(self,obsmat):
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

        assignment = pyro.sample('assignment', dist.Categorical(weights))

        for r in np.arange(obsmat.shape[0]):
            rowind = obsmat[r,1].type(torch.long)
            colind = obsmat[r,2].type(torch.long)
            d = dist.Beta(alphas[assignment,rowind,colind],betas[assignment,rowind,colind])
            pyro.sample('obs_{}'.format(r),d,obs = obsmat[r,0])

    def classifier_multi_obs(self, obsmat, temperature): # temperature = 1 to sample
        inferred_model = infer_discrete(self.trained_model_multi, temperature=temperature,
                                        first_available_dim=-1)  # avoid conflict with data plate
        trace = poutine.trace(inferred_model).get_trace(obsmat)
        return trace.nodes["assignment"]["value"]

    def group_classify(self,fitted_model,test_data):
        # hard and soft classification on group models
        guide = getattr(fitted_model,'guide')
        train_data = getattr(fitted_model,'data')
        guide_trace = poutine.trace(guide).get_trace(train_data)  # record the globals
        self.trained_model_multi = poutine.replay(self.model_multi_obs_grp_raw, trace = guide_trace)
        # initialize storage
        niter = 500
        stor_grp = torch.empty(size = test_data.shape)
        stor_grp_prb = torch.empty(size = [test_data.shape[0],test_data.shape[1],test_data.shape[2],self.K])
        # iterate and classify nested person -
        for i in range(test_data.shape[0]):
            print('group learning, onto p{}'.format(i))
            for j in range(test_data.shape[1]):
                for k in range(test_data.shape[2]):
                    # print('group learning, onto p{} r{} c{}'.format(i,j,k))
                    obsmat = torch.tensor([test_data[i,j,k],j,k]).float().unsqueeze(0)
                    # first MAP classification
                    grp = self.classifier_multi_obs(obsmat, temperature = 0)
                    print('here')
                    stor_grp[i,j,k] = grp
                    # second use sampling to get group prob
                    stor = torch.zeros(niter)
                    for it in np.arange(niter):
                        stor[it] = self.classifier_multi_obs(obsmat, temperature = 1)
                    grp_prb = [(stor == i).sum()/float(len(stor)) for i in np.arange(self.K)]
                    stor_grp_prb[i,j,k,:] = torch.tensor(grp_prb)

        self.stor_grp = stor_grp
        self.stor_grp_prb = stor_grp_prb
        # self.stor_grp_prb = stor_grp

    def group_classify_single_subject(self,i,subject_data,l):
        print('onto subject {}'.format(i))
        niter = 500
        for j in range(subject_data.shape[0]):
            for k in range(subject_data.shape[1]):
                obsmat = torch.tensor([subject_data[j,k],j,k]).unsqueeze(0)
                # first MAP classification
                grp = self.classifier_multi_obs(obsmat, temperature = 0)
                # second use sampling to get group prob
                stor = torch.zeros(niter)
                for it in np.arange(niter):
                    stor[it] = self.classifier_multi_obs(obsmat, temperature = 1)
                grp_prb = [(stor == i).sum()/float(len(stor)) for i in np.arange(self.K)]
                # queue.put((i, j, k, grp.clone(), torch.tensor(grp_prb).detach().clone()))
                l.append((i, j, k, grp.clone(), torch.tensor(grp_prb).clone()))
        print('done classifying {}'.format(i))

    def group_classify_parallel(self, fitted_model, test_data):
        # hard and soft classification on group models
        guide = getattr(fitted_model,'guide')
        train_data = getattr(fitted_model,'data')
        guide_trace = poutine.trace(guide).get_trace(train_data)  # record the globals
        self.trained_model_multi = poutine.replay(self.model_multi_obs_grp_raw, trace = guide_trace)
        # initialize storage
        stor_grp = torch.empty(size = test_data.shape)
        stor_grp_prb = torch.empty(size = [test_data.shape[0],test_data.shape[1],test_data.shape[2],self.K])
        ncpu = 5
        narray = np.floor(test_data.shape[0]/ncpu) + 1
        participant_chunks = np.array_split(list(range(test_data.shape[0])), narray)
        # print(participant_chunks)
        for chunk in participant_chunks:
            with multiprocessing.Manager() as manager:
                q = manager.list()
                p = [multiprocessing.Process(target = self.group_classify_single_subject, args = (i, test_data[i,:,:], q)) for i in chunk]
                [i.start() for i in p]
                [i.join() for i in p]
                # read content of manager.list into array storage
                for tp in q:
                    i,j,k = tp[0:3] # first 3 elements of each returned tuple are indices
                    stor_grp[i,j,k] = tp[3]
                    stor_grp_prb[i,j,k,:] = tp[4]
        self.stor_grp = stor_grp
        self.stor_grp_prb = stor_grp_prb
        print(stor_grp[0,:,:])
        print('DONE CLASSIFYING')

    def group_infer(self,fitted_model):
        # make stimuliselector
        stmslct = StimuliSelector()
        # generate inference hard and soft
        [self.hdist, self.hmeans] = stmslct.hard_grp_raw_infer_unobserved(self.stor_grp, getattr(fitted_model, 'map_estimates'))
        [self.sdist, self.smeans] = stmslct.soft_grp_raw_infer_unobserved(self.stor_grp_prb, getattr(fitted_model, 'map_estimates'))
        # [self.sdist, self.smeans] = stmslct.hard_grp_raw_infer_unobserved(self.stor_grp_prb, getattr(fitted_model, 'map_estimates'))
        return self.hdist, self.hmeans, self.sdist, self.smeans

    def group_compute_metrics(self, test_data):
        ### compute three metrics: absolute error, squared error, and log prob
        # initialzie storage
        self.h_ae = np.empty(self.hmeans.shape, dtype = 'object')
        self.s_ae = np.empty(self.smeans.shape, dtype = 'object')
        self.h_se = np.empty(self.hmeans.shape, dtype = 'object')
        self.s_se = np.empty(self.smeans.shape, dtype = 'object')
        self.h_lp = np.empty(self.hdist.shape, dtype = 'object')
        self.s_lp = np.empty(self.sdist.shape, dtype = 'object')
        for i in range(test_data.shape[0]):
            for j in range(test_data.shape[1]):
                for k in range(test_data.shape[2]):
                    # print(self.smeans[i,j,k])
                    # print(test_data[i])
                    # absolute error
                    self.h_ae[i,j,k] = torch.abs(self.hmeans[i,j,k] - test_data[i]).detach()
                    self.s_ae[i,j,k] = torch.abs(self.smeans[i,j,k] - test_data[i]).detach()
                    # squared error
                    self.h_se[i,j,k] = torch.mul(self.h_ae[i,j,k], self.h_ae[i,j,k]).detach()
                    self.s_se[i,j,k] = torch.mul(self.s_ae[i,j,k], self.s_ae[i,j,k]).detach()
                    #log prob
                    print(self.hdist[i,j,k].sample())
                    self.h_lp[i,j,k] = self.hdist[i,j,k].log_prob(test_data[i]).detach()
                    self.s_lp[i,j,k] = self.sdist[i,j,k].log_prob(test_data[i]).detach()

        return self.h_ae, self.s_ae, self.h_se, self.s_se, self.h_lp, self.s_lp

    def group_compute_metrics_parallel(self, test_data, split_id, ml):
        # still doesn't work, too much data for the manager to spawn new processes
        # parallelize at the person-inference level instead, use group_classify_parallel
        # immediately return to queue instead of storing all
        for i in range(test_data.shape[0]):
            for j in range(test_data.shape[1]):
                for k in range(test_data.shape[2]):
                    # absolute
                    h_ae = torch.abs(self.hmeans[i,j,k] - test_data[i]).detach()
                    s_ae = torch.abs(self.smeans[i,j,k] - test_data[i]).detach()
                    # squared
                    h_se = torch.mul(h_ae, h_ae).detach()
                    s_se = torch.mul(s_ae, s_ae).detach()
                    #log prob
                    h_lp = self.hdist[i,j,k].log_prob(test_data[i]).detach()
                    s_lp = self.sdist[i,j,k].log_prob(test_data[i]).detach()
                    # construct returned tuple
                    ret = (split_id, i, j , k, h_ae, s_ae, h_se, s_se, h_lp, s_lp)
                    ml.append(ret)

    @config_enumerate
    def model_multi_obs_dim_raw(self,obsmat):
        num_topics = self.K
        # This is a reasonable prior for dirichlet concentrations
        gamma_prior = dist.Gamma(
            2 * torch.ones(self.nfeatures, self.ncol),
            1/3 * torch.ones(self.nfeatures, self.ncol)
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
    def dimension_new_guide(self,obsmat):
        # These are just the previous values we can use to initialize params here
        map_estimates = getattr(self.fitted_model,'map_estimates')
        initial_topic_weights = map_estimates['topic_weights']
        initial_alpha = map_estimates['topic_weights']
        initial_topic_a = map_estimates['topic_a']
        initial_topic_b =map_estimates['topic_b']

        # Use poutine.block to Keep our learned values of global parameters.
        with poutine.block(hide_types=["param"]):
            # This has to match the structure of the model
            with pyro.plate('topic', self.K):
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

    def initialize_multi_obs_dim(self, seed, obsmat):
        pyro.set_rng_seed(seed)
        if 'new_participant_topic_q' in pyro.get_param_store().keys():
            pyro.get_param_store().__delitem__('new_participant_topic_q')
        self.svi = SVI(self.model_multi_obs_dim_raw, self.dimension_new_guide, self.optim, loss = self.elbo)
        return self.svi.loss(self.model_multi_obs_dim_raw, self.dimension_new_guide, obsmat)

    def dimension_learn(self,fitted_model,test_data):
        self.fitted_model = fitted_model
        stor_dim_prb = torch.empty(size = [test_data.shape[0],test_data.shape[1],test_data.shape[2],self.K])
        for i in range(test_data.shape[0]):
            print('dimension learning, onto p{}'.format(i))
            for j in range(test_data.shape[1]):
                for k in range(test_data.shape[2]):
                    # print('dimension learning, onto p{} r{} c{}'.format(i,j,k))
                    obsmat = torch.tensor([test_data[i,j,k],j,k]).float().unsqueeze(0)
                    loss, seed = min((self.initialize_multi_obs_dim(seed,obsmat),
                                      seed) for seed in range(100))
                    self.initialize_multi_obs_dim(seed,obsmat)
                    tik = time.time()
                    for s in range(self.sparse_nstep):
                        # if s % 1000 == 0:
                        #     print(s//1000)
                        loss = self.svi.step(obsmat)
                    print('time! {}'.format(time.time() - tik))
                    stor_dim_prb[i,j,k,:] = pyro.get_param_store()['new_participant_topic_q']
        self.stor_dim_prb = stor_dim_prb
        return self.stor_dim_prb

    def dimension_learn_subset(self,fitted_model,test_data, stim_size = 700):
        # reproducibility is currently not achieved (want randomness)
        self.fitted_model = fitted_model
        stor_dim_prb = torch.empty(size = [test_data.shape[0],test_data.shape[1],test_data.shape[2],self.K])
        enu = np.ndenumerate(torch.empty(size = [test_data.shape[0],test_data.shape[1],test_data.shape[2]]))
        enu = [i[0] for i in enu]
        inds = random.sample(enu, stim_size)
        self.stim_subset = inds
        for ind in inds:
            print('onto {}'.format(ind))
            [i,j,k] = [ind[0], ind[1], ind[2]]
            # print('dimension learning, onto p{} r{} c{}'.format(i,j,k))
            obsmat = torch.tensor([test_data[i,j,k],j,k]).float().unsqueeze(0)
            loss, seed = min((self.initialize_multi_obs_dim(seed,obsmat),
                              seed) for seed in range(100))
            self.initialize_multi_obs_dim(seed,obsmat)
            tik = time.time()
            for s in range(self.sparse_nstep):
                # if s % 1000 == 0:
                #     print(s//1000)
                loss = self.svi.step(obsmat)
            print('time! {}'.format(time.time() - tik))
            stor_dim_prb[i,j,k,:] = pyro.get_param_store()['new_participant_topic_q']
        self.stor_dim_prb = stor_dim_prb
        return self.stor_dim_prb

    def dimension_learn_single_subject(self,i,subject_data,l):
        print('dimension learning, onto p{}'.format(i))
        for j in range(subject_data.shape[0]):
            for k in range(subject_data.shape[1]):
                # print('dimension learning, onto p{} r{} c{}'.format(i,j,k))
                obsmat = torch.tensor([subject_data[j,k],j,k]).float().unsqueeze(0)
                loss, seed = min((self.initialize_multi_obs_dim(seed,obsmat),
                                  seed) for seed in range(100))
                self.initialize_multi_obs_dim(seed,obsmat)
                tik = time.time()
                for s in range(2000):
                    if s % 1000 == 0:
                        print(s//1000)
                    loss = self.svi.step(obsmat)
                print('time! {}'.format(time.time()))
                out = pyro.get_param_store()['new_participant_topic_q'].detach()
                l.append((i, j, k, out.clone()))
        print('done learning')

    def dimension_learn_parallel(self, fitted_model, test_data):
        self.fitted_model = fitted_model
        stor_dim_prb = torch.empty(size = [test_data.shape[0],test_data.shape[1],test_data.shape[2],self.K])
        #
        ncpu = 5
        narray = np.floor(test_data.shape[0]/ncpu) + 1
        participant_chunks = np.array_split(list(range(test_data.shape[0])), narray)
        # print(participant_chunks)
        for chunk in participant_chunks:
            with multiprocessing.Manager() as manager:
                q = manager.list()
                p = [multiprocessing.Process(target = self.dimension_learn_single_subject, args = (i, test_data[i,:,:], q)) for i in chunk]
                [i.start() for i in p]
                [i.join() for i in p]
                # read content of manager.list into array storage
                for tp in q:
                    i,j,k = tp[0:3] # first 3 elements of each returned tuple are indices
                    stor_dim_prb[i,j,k,:] = tp[4]
        self.stor_dim_prb = stor_dim_prb
        print('DONE LEARNING')

    def dimension_infer(self, fitted_model):
        # make stimuliselector
        stmslct = StimuliSelector()
        # generate inference hard and soft
        [self.ddist, self.dmeans] = stmslct.dim_raw_infer_unobserved(self.stor_dim_prb, getattr(fitted_model, 'map_estimates'))
        return self.ddist, self.dmeans

    def dimension_compute_metrics(self, test_data):
        ### compute three metrics: absolute error, squared error, and log prob
        # initialzie storage
        self.ae = np.empty(self.dmeans.shape, dtype = 'object')
        self.se = np.empty(self.dmeans.shape, dtype = 'object')
        self.lp = np.empty(self.ddist.shape, dtype = 'object')

        for i in range(test_data.shape[0]):
            for j in range(test_data.shape[1]):
                for k in range(test_data.shape[2]):
                    # absolute error
                    self.ae[i,j,k] = torch.abs(self.dmeans[i,j,k] - test_data[i]).detach()
                    # squared error
                    self.se[i,j,k] = torch.mul(self.ae[i,j,k], self.ae[i,j,k]).detach()
                    #log prob
                    self.lp[i,j,k] = self.ddist[i,j,k].log_prob(test_data[i]).detach()
        return self.ae, self.se, self.lp

    def dimension_compute_metrics_parallel(self, test_data, split_id, ml):
        # immediately return to queue instead of storing all
        for i in range(test_data.shape[0]):
            for j in range(test_data.shape[1]):
                for k in range(test_data.shape[2]):
                    # absolute
                    d_ae = torch.abs(self.dmeans[i,j,k] - test_data[i]).detach()
                    # squared
                    d_se = torch.mul(d_ae, d_ae).detach()
                    #log prob
                    d_lp = self.hdist[i,j,k].log_prob(test_data[i]).detach()
                    # construct returned tuple
                    ret = (split_id, i, j , k, d_ae, d_se, d_lp)
                    ml.append(ret)


class ModelEvaluator():
    '''
    k fold split
    for each split:
        train a model with object level params on the train set
        construct a stimuliselector
        for each person in the test set
            use each cell as input for single stim generation
            (predicting the rest of the matrix with a single cell as input)
            calculate the error term on the predictions (absolute?)
    '''
    def __init__(self, data, target, dtype, auto, mtype, maxk = 3, nfold = 5, random_state = None):
        self.data = data
        self.maxk = maxk
        self.target = target
        self.dtype = dtype
        self.auto = auto
        self.mtype = mtype
        self.nfold = nfold
        nseeds = maxk + 1 # total number of random seeds needed: 1 for splitting, 1 for each train-test procedure
        if random_state:
            self.random_seeds = list(range(random_state, random_state+nseeds))
        else:
            self.random_seeds = [random.randint(0,999999) for i in range(nseeds)]
        self.rseed_counter = 0
        # make train-test split, split at the PERSON level
        kf = KFold(n_splits = self.nfold, shuffle = True, random_state = self.random_seeds[self.rseed_counter])
        self.rseed_counter += 1;
        self.split = kf.split(list(range(self.data.shape[0])))

    def evaluate(self):

        for i in range(self.maxk):
            random.seed(self.random_seeds[self.rseed_counter])
            self.rseed_counter += 1
            k = i + 1

            for train_index, test_index in self.split:
                train = self.data[train_index,:,:]
                test = self.data[test_index,:,:]
                # first fit the model
                mdl = TransitionModel(
                    train, k, self.target, self.dtype, self.auto, self.mtype, stickbreak = False
                )
                mdl.fit()

                mdl_multi_obs = SparseModel(
                    train, k, self.target, self.dtype, self.auto, self.mtype, stickbreak = False
                )
                if 'gr' in self.mtype:
                    mdl_multi_obs.group_classify(mdl,test)
                    hdist, hmeans, sdist, smeans = mdl_multi_obs.group_infer(mdl)
                    self.h_ae, self.s_ae, self.h_se, self.s_se, self.h_lp, self.s_lp = mdl_multi_obs.group_compute_metrics(test)
                    return self.h_ae, self.s_ae, self.h_se, self.s_se, self.h_lp, self.s_lp

                elif 'dim' in self.mtype:
                    mdl_multi_obs.dimension_learn(mdl,test)
                    ddist, dmeans = mdl_multi_obs.dimension_infer(mdl)
                    self.d_ae, self.d_se, self.d_lp = mdl_multi_obs.dimension_compute_metrics(test)
                    return self.d_ae, self.d_se, self.d_lp


class ParallelEvaluator():
    #### two levels of parallelization: the outer level done by submitting multiple slurm jobs
    def __init__(self, data, target, dtype, auto, mtype, maxk, K, nfold = 5, random_state = None, sparse_nstep = 10000):
        self.data = data
        self.maxk = maxk
        self.K = K
        self.target = target
        self.dtype = dtype
        self.auto = auto
        self.mtype = mtype
        self.nfold = nfold
        nseeds = maxk + 1 # total number of random seeds needed: 1 for splitting, 1 for each train-test procedure
        if random_state:
            self.random_seeds = list(range(random_state, random_state+nseeds))
        else:
            self.random_seeds = [random.randint(0,999999) for i in range(nseeds)]
        self.rseed_counter = 0
        # make train-test split, split at the PERSON level
        kf = KFold(n_splits = self.nfold, shuffle = True, random_state = self.random_seeds[self.rseed_counter])
        self.rseed_counter += 1;
        self.split = kf.split(list(range(self.data.shape[0])))

        self.sparse_nstep = sparse_nstep

    def evaluate_singleK(self, split, split_id, queue): # multiprocessing.Queue arg to store output
        print('doing a split')
        random.seed(self.random_seeds[self.K])

        train = self.data[split[0],:,:]
        test = self.data[split[1],:,:]

        # first fit the model
        k = self.K # note the absences of +1 - make sure in range 1,10
        mdl = TransitionModel(
            train, k, self.target, self.dtype, self.auto, self.mtype, stickbreak = False
        )
        mdl.fit()

        mdl_multi_obs = SparseModel(
            train, k, self.target, self.dtype, self.auto, self.mtype, stickbreak = False
        )
        if 'gr' in self.mtype:
            mdl_multi_obs.group_classify(mdl,test)
            # hdist, hmeans, sdist, smeans = mdl_multi_obs.group_infer(mdl)
            # mdl_multi_obs.group_compute_metrics_parallel(test,split_id,queue)

        elif 'dim' in self.mtype:
            mdl_multi_obs.dimension_learn(mdl,test)
            ddist, dmeans = mdl_multi_obs.dimension_infer(mdl)
            mdl_multi_obs.dimension_compute_metrics_parallel(test,split_id,queue)

    def evaluate_parallel(self):
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     splits = [(train,test) for train, test in self.split]
        #     results = [executor.submit(self.evaluate_singleK,i) for i in splits]
        splits = [(train,test) for train, test in self.split]
        # q = multiprocessing.SimpleQueue()
        # p = [multiprocessing.Process(target = self.evaluate_singleK,args = (splits[i], i, q)) for i in range(len(splits))]
        # [i.start() for i in p]
        # # get output out of q
        # q_out = []
        # print('here1')
        # while 1:
        #     running = any(i.is_alive() for i in p)
        #     while not q.empty():
        #        q_out.append(q.get())
        #     if not running:
        #         break
        #     time.sleep(1)
        #
        # [i.join() for i in p]
        # # get output out of q
        # print('joined')
        #
        # # pool_out = pool.map(self.evaluate_singleK, splits)
        #
        # print('\n\n\nPOOL OUT\n\n\n')
        # print('q:\n{}'.format(q_out))
        #
        # return q_out

    def evaluate_parallel_alt(self,split_id = None):
        if split_id:
            splits = (list(self.split)[split_id-1],) # exogenously make sure split ids and nfold match up. split id 1 index
            print('MANUAL PARALLEL just doing one split')
        else:
            splits = list(self.split)
            grp_out = []

        for splt in splits:
            train_index = splt[0]
            test_index = splt[1]
            print('doing a split')
            seed = self.random_seeds[self.K]
            if split_id:
                seed += split_id
            random.seed(seed) # means that for dim

            train = self.data[train_index,:,:]
            test = self.data[test_index,:,:]
            k = self.K
            # first fit the model
            mdl = TransitionModel(
                train, k, self.target, self.dtype, self.auto, self.mtype, stickbreak = False
            )
            mdl.fit()

            mdl_multi_obs = SparseModel(
                train, k, self.target, self.dtype, self.auto, self.mtype, stickbreak = False
            )

            if 'dim' in self.mtype:
                # mdl_multi_obs.dimension_learn_parallel(mdl,test) # doesn't work for dimensional because of autograd threading
                mdl_multi_obs.dimension_learn(mdl,test)
                ddist, dmeans = mdl_multi_obs.dimension_infer(mdl)
                self.d_ae, self.d_se, self.d_lp = mdl_multi_obs.dimension_compute_metrics(test)
                return (split_id, self.d_ae, self.d_se, self.d_lp) # note that dim must be run with a split id (manual parallelization)

            elif 'gr' in self.mtype:
                mdl_multi_obs.group_classify_parallel(mdl,test)
                hdist, hmeans, sdist, smeans = mdl_multi_obs.group_infer(mdl)
                self.h_ae, self.s_ae, self.h_se, self.s_se, self.h_lp, self.s_lp = mdl_multi_obs.group_compute_metrics(test)
                grp_out.append((self.h_ae, self.s_ae, self.h_se, self.s_se, self.h_lp, self.s_lp))
        return grp_out

    def evaluate_parallel_alt_subset(self,split_id = None, stim_size = 700):
        if split_id:
            splits = (list(self.split)[split_id-1],) # exogenously make sure split ids and nfold match up. split id 1 index
            print('MANUAL PARALLEL just doing one split')
        else:
            splits = list(self.split)
            grp_out = []

        for splt in splits:
            train_index = splt[0]
            test_index = splt[1]
            print('doing a split')
            seed = self.random_seeds[self.K]
            if split_id:
                seed += split_id
            random.seed(seed) # means that for dim

            train = self.data[train_index,:,:]
            test = self.data[test_index,:,:]
            k = self.K
            # first fit the model
            mdl = TransitionModel(
                train, k, self.target, self.dtype, self.auto, self.mtype, stickbreak = False
            )
            mdl.fit()

            mdl_multi_obs = SparseModel(
                train, k, self.target, self.dtype, self.auto, self.mtype, stickbreak = False, sparse_nstep = self.sparse_nstep
            )

            if 'dim' in self.mtype:
                # mdl_multi_obs.dimension_learn_parallel(mdl,test) # doesn't work for dimensional because of autograd threading
                mdl_multi_obs.dimension_learn_subset(mdl,test,stim_size = stim_size)
                ddist, dmeans = mdl_multi_obs.dimension_infer(mdl)
                self.d_ae, self.d_se, self.d_lp = mdl_multi_obs.dimension_compute_metrics(test)
                stim_subset = getattr(mdl_multi_obs,'stim_subset')# document the actually selected stims to index the prediction matrices
                return (split_id, self.d_ae, self.d_se, self.d_lp, stim_subset) # note that dim must be run with a split id (manual parallelization)

            elif 'gr' in self.mtype:
                mdl_multi_obs.group_classify_parallel(mdl,test)
                hdist, hmeans, sdist, smeans = mdl_multi_obs.group_infer(mdl)
                self.h_ae, self.s_ae, self.h_se, self.s_se, self.h_lp, self.s_lp = mdl_multi_obs.group_compute_metrics(test)
                grp_out.append((self.h_ae, self.s_ae, self.h_se, self.s_se, self.h_lp, self.s_lp))
        return grp_out



if __name__ == '__main__':
    # import pickled data
    import pickle
    import time
    import sys
    # with open('/u/zidong/data/tomtom_data_preprocessed_withadded.pkl','rb') as f:
    # with open('C:/Users/zhaoz/group-inference/data/tomtom_data_preprocessed_withadded.pkl','rb') as f:
    #     [tself_norm_all_3d, tself_norm_noauto_3d, tself_raw_all_3d, tself_raw_noauto_3d,
    #     ttarg_norm_all_3d, ttarg_norm_noauto_3d, ttarg_raw_all_3d, ttarg_raw_noauto_3d,
    #     tavg_norm_all_3d, tavg_norm_noauto_3d, tavg_raw_all_3d, tavg_raw_noauto_3d] = pickle.load(f)
    #
    # trunc_data = tself_raw_noauto_3d[:,:,:] # truncated data for quick debug
    #
    # phn = ParallelEvaluator(trunc_data, 'self','raw','noauto','grp', maxk = 10, K = 3, random_state = 888)
    # phn.evaluate_parallel_alt()

    for i in range(10):
        print(i)
        [print(i) for i in range(5)]


    # tomtom = TransitionModel(
    #     data = tself_norm_all_3d,
    #     K = 2,
    #     target = 'self',
    #     dtype = 'norm',
    #     auto = 'all',
    #     mtype = 'dim'
    # )
    #
    # tomtom.fit()

    # # # testing how np.ndenumerate works
    # with open('C:/Users/zhaoz/group-inference/data/generated/stor_grp_1feat.pkl','rb') as f:
    #     [hard1, soft1] = pickle.load(f)
    # with open('C:/Users/zhaoz/group-inference/data/generated/stor_grp_3feat.pkl','rb') as f:
    #     [hard3, soft3] = pickle.load(f)
    # # a = np.ndenumerate(soft3)
    # # b = np.ndenumerate(soft3.sum(axis=len(soft3.shape)-1))
    # # for i in range(5):
    # #     nde = next(b)
    # #     ndei = nde[0]
    # #     print(soft3[ndei])
    #
    # # # testing how MixtureSameFamily work with torch.distributions shapes
    # # da = dist.Beta(torch.tensor([[1,1],[1,1]]),torch.tensor([[.2,.3],[.4,.5]]))
    # # db = dist.Beta(torch.tensor([5,5,5,5]),torch.tensor([.2,.3,.4,.5]))
    # # print(f'batch size {da.batch_shape}')
    # # print(f'event size {da.event_shape}')
    # # mix_alt = dist.Categorical(torch.tensor([1,2,3,4]))
    # # mix_alt2 = dist.Categorical(torch.tensor([1,2]))
    # # dc = torch.distributions.mixture_same_family.MixtureSameFamily(mix_alt2, da.to_event(1))
    # # tik = time.time()
    # # print(sys.getsizeof(dc))
    # # print(dc.sample(torch.Size([1])))
    # # print(dc.mean)
    # #
    # # print(time.time() - tik)
    # #
    # # dtest = dist.Beta(torch.tensor([[1,2,3],[4,5,6]]), torch.tensor([[.1,.1,.1],[.1,.1,.1]]))
    # # print(dtest.to_event(1).batch_shape, dtest.to_event(1).event_shape)
    # # print(type(dtest.to_event(1)))
    # # mixtest = dist.Categorical(torch.tensor([1,1]))
    # # d = torch.distributions.mixture_same_family.MixtureSameFamily(mixtest, dtest.to_event(1))
    # # print(d.batch_shape, d.event_shape)
    # # d.sample(torch.tensor([1]))
    # # print(d.mean)
    #
    # # # testing how tensor.permute works
    # # a = torch.tensor([[[1,2],[3,4]],[[5,6],[7,8]],[[9,10],[11,12]]])
    # # b = a.permute(2,0,1)
    # # print(a[0])
    # # print(b[:,0,:])
    #
    # with open('C:/Users/zhaoz/group-inference/data/generated/tomtom_fitted_models.pkl','rb') as f:
    #     [seeds_self_norm_all_grp,maps_self_norm_all_grp,logprobs_self_norm_all_grp,mems_self_norm_all_grp,
    #      seeds_self_norm_all_dim,maps_self_norm_all_dim,logprobs_self_norm_all_dim,
    #      seeds_self_norm_noauto_grp,maps_self_norm_noauto_grp,logprobs_self_norm_noauto_grp,mems_self_norm_noauto_grp,
    #      seeds_self_norm_noauto_dim,maps_self_norm_noauto_dim,logprobs_self_norm_noauto_dim,
    #      seeds_self_raw_all_grp,maps_self_raw_all_grp,logprobs_self_raw_all_grp,mems_self_raw_all_grp,
    #      seeds_self_raw_all_dim,maps_self_raw_all_dim,logprobs_self_raw_all_dim,
    #      seeds_self_raw_noauto_grp,maps_self_raw_noauto_grp,logprobs_self_raw_noauto_grp,mems_self_raw_noauto_grp,
    #      seeds_self_raw_noauto_dim,maps_self_raw_noauto_dim,logprobs_self_raw_noauto_dim] = pickle.load(f)
    #
    # tik = time.time()
    # # # print(maps_self_raw_noauto_grp[2]['alpha'].shape)
    # # a = maps_self_raw_noauto_grp[2]['alpha']
    # # b = maps_self_raw_noauto_grp[2]['beta']
    # # all_dist = dist.Beta(a,b)
    # # all_dist = all_dist.to_event(2)
    # # print(all_dist.batch_shape, all_dist.event_shape)
    # # mix = dist.Categorical(torch.tensor([1,1,1]))
    # #
    # # c = torch.distributions.mixture_same_family.MixtureSameFamily(mix,all_dist)
    # #
    # # print(c.sample(torch.Size([5])))
    # # print(c.sample(torch.Size([5])).shape)
    # # print(c.mean)
    # # print(time.time()-tik)
    # # jk = np.array([[c,c],[a,b]])
    # # print(jk.dtype)
    #
    # stimselect = StimuliSelector()
    #
    # [h1dists,h1means] = stimselect.hard_grp_raw_infer_unobserved(hard1,maps_self_raw_noauto_grp[2])
    # print(h1means[0])
    # print(h1means.shape)
    # [s1dists,s1means] = stimselect.soft_grp_raw_infer_unobserved(soft1, maps_self_raw_noauto_grp[2])
    # tok = time.time()
    # print(tok - tik)
    # # save inferred
    # with open('tomtom_sparse_inference_grp_1feat.pkl','wb') as f:
    #     pickle.dump([h1dists,h1means,s1dists,s1means],f)
    #
    #
    # [h3dists,h3means] = stimselect.hard_grp_raw_infer_unobserved(hard3,maps_self_raw_noauto_grp[2])
    # print(h3means[0])
    # print(h3means.shape)
    # [s3dists,s3means] = stimselect.soft_grp_raw_infer_unobserved(soft3, maps_self_raw_noauto_grp[2])
    # tok = time.time()
    # print(tok - tik)
    # # save inferred
    # # with open('tomtom_sparse_inference_grp_3feat.pkl','wb') as f:
    # #     pickle.dump([h3dists,h3means,s3dists,s3means],f)
    # import joblib
    # joblib.dump([h3dists,h3means,s3dists,s3means],'tomtom_sparse_inference_grp_3feat.z') # ended up running on cluster, needed 1000G mem to save
    #
    #
    #
    #
    #
    #
    # print(time.time() - tok)
