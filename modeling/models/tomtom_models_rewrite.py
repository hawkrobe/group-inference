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
            if 'grp' in self.mtype:
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

if __name__ == '__main__':
    # import pickled data
    import pickle
    import time
    import sys
    # with open('C:/Users/zhaoz/group-inference/data/tomtom_data_preprocessed.pkl','rb') as f:
    #     [tself_norm_all_3d, tself_norm_noauto_3d, tself_raw_all_3d, tself_raw_noauto_3d,
    #     ttarg_norm_all_3d, ttarg_norm_noauto_3d, ttarg_raw_all_3d, ttarg_raw_noauto_3d,
    #     tavg_norm_all_3d, tavg_norm_noauto_3d, tavg_raw_all_3d, tavg_raw_noauto_3d] = pickle.load(f)

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

    # # testing how np.ndenumerate works
    with open('C:/Users/zhaoz/group-inference/data/generated/stor_grp_1feat.pkl','rb') as f:
        [hard1, soft1] = pickle.load(f)
    with open('C:/Users/zhaoz/group-inference/data/generated/stor_grp_3feat.pkl','rb') as f:
        [hard3, soft3] = pickle.load(f)
    # a = np.ndenumerate(soft3)
    # b = np.ndenumerate(soft3.sum(axis=len(soft3.shape)-1))
    # for i in range(5):
    #     nde = next(b)
    #     ndei = nde[0]
    #     print(soft3[ndei])

    # # testing how MixtureSameFamily work with torch.distributions shapes
    # da = dist.Beta(torch.tensor([[1,1],[1,1]]),torch.tensor([[.2,.3],[.4,.5]]))
    # db = dist.Beta(torch.tensor([5,5,5,5]),torch.tensor([.2,.3,.4,.5]))
    # print(f'batch size {da.batch_shape}')
    # print(f'event size {da.event_shape}')
    # mix_alt = dist.Categorical(torch.tensor([1,2,3,4]))
    # mix_alt2 = dist.Categorical(torch.tensor([1,2]))
    # dc = torch.distributions.mixture_same_family.MixtureSameFamily(mix_alt2, da.to_event(1))
    # tik = time.time()
    # print(sys.getsizeof(dc))
    # print(dc.sample(torch.Size([1])))
    # print(dc.mean)
    #
    # print(time.time() - tik)
    #
    # dtest = dist.Beta(torch.tensor([[1,2,3],[4,5,6]]), torch.tensor([[.1,.1,.1],[.1,.1,.1]]))
    # print(dtest.to_event(1).batch_shape, dtest.to_event(1).event_shape)
    # print(type(dtest.to_event(1)))
    # mixtest = dist.Categorical(torch.tensor([1,1]))
    # d = torch.distributions.mixture_same_family.MixtureSameFamily(mixtest, dtest.to_event(1))
    # print(d.batch_shape, d.event_shape)
    # d.sample(torch.tensor([1]))
    # print(d.mean)

    # # testing how tensor.permute works
    # a = torch.tensor([[[1,2],[3,4]],[[5,6],[7,8]],[[9,10],[11,12]]])
    # b = a.permute(2,0,1)
    # print(a[0])
    # print(b[:,0,:])

    with open('C:/Users/zhaoz/group-inference/data/generated/tomtom_fitted_models.pkl','rb') as f:
        [seeds_self_norm_all_grp,maps_self_norm_all_grp,logprobs_self_norm_all_grp,mems_self_norm_all_grp,
         seeds_self_norm_all_dim,maps_self_norm_all_dim,logprobs_self_norm_all_dim,
         seeds_self_norm_noauto_grp,maps_self_norm_noauto_grp,logprobs_self_norm_noauto_grp,mems_self_norm_noauto_grp,
         seeds_self_norm_noauto_dim,maps_self_norm_noauto_dim,logprobs_self_norm_noauto_dim,
         seeds_self_raw_all_grp,maps_self_raw_all_grp,logprobs_self_raw_all_grp,mems_self_raw_all_grp,
         seeds_self_raw_all_dim,maps_self_raw_all_dim,logprobs_self_raw_all_dim,
         seeds_self_raw_noauto_grp,maps_self_raw_noauto_grp,logprobs_self_raw_noauto_grp,mems_self_raw_noauto_grp,
         seeds_self_raw_noauto_dim,maps_self_raw_noauto_dim,logprobs_self_raw_noauto_dim] = pickle.load(f)

    tik = time.time()
    # # print(maps_self_raw_noauto_grp[2]['alpha'].shape)
    # a = maps_self_raw_noauto_grp[2]['alpha']
    # b = maps_self_raw_noauto_grp[2]['beta']
    # all_dist = dist.Beta(a,b)
    # all_dist = all_dist.to_event(2)
    # print(all_dist.batch_shape, all_dist.event_shape)
    # mix = dist.Categorical(torch.tensor([1,1,1]))
    #
    # c = torch.distributions.mixture_same_family.MixtureSameFamily(mix,all_dist)
    #
    # print(c.sample(torch.Size([5])))
    # print(c.sample(torch.Size([5])).shape)
    # print(c.mean)
    # print(time.time()-tik)
    # jk = np.array([[c,c],[a,b]])
    # print(jk.dtype)

    stimselect = StimuliSelector()

    [h1dists,h1means] = stimselect.hard_grp_raw_infer_unobserved(hard1,maps_self_raw_noauto_grp[2])
    print(h1means[0])
    print(h1means.shape)
    [s1dists,s1means] = stimselect.soft_grp_raw_infer_unobserved(soft1, maps_self_raw_noauto_grp[2])
    tok = time.time()
    print(tok - tik)
    # save inferred
    with open('tomtom_sparse_inference_grp_1feat.pkl','wb') as f:
        pickle.dump([h1dists,h1means,s1dists,s1means],f)


    [h3dists,h3means] = stimselect.hard_grp_raw_infer_unobserved(hard3,maps_self_raw_noauto_grp[2])
    print(h3means[0])
    print(h3means.shape)
    [s3dists,s3means] = stimselect.soft_grp_raw_infer_unobserved(soft3, maps_self_raw_noauto_grp[2])
    tok = time.time()
    print(tok - tik)
    # save inferred
    # with open('tomtom_sparse_inference_grp_3feat.pkl','wb') as f:
    #     pickle.dump([h3dists,h3means,s3dists,s3means],f)
    import joblib
    joblib.dump([h3dists,h3means,s3dists,s3means],'tomtom_sparse_inference_grp_3feat.z') # ended up running on cluster, needed 1000G mem to save






    print(time.time() - tok)
