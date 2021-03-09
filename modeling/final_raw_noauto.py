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

import models.tomtom_models_rewrite as tm
import pickle


if __name__ == '__main__':
    # with open('C:/Users/zhaoz/group-inference/data/tomtom_data_preprocessed_withadded.pkl','rb') as f:
    with open('tomtom_data_preprocessed_withadded.pkl','rb') as f:
        [tself_norm_all_3d, tself_norm_noauto_3d, tself_raw_all_3d, tself_raw_noauto_3d,
        ttarg_norm_all_3d, ttarg_norm_noauto_3d, ttarg_raw_all_3d, ttarg_raw_noauto_3d,
        tavg_norm_all_3d, tavg_norm_noauto_3d, tavg_raw_all_3d, tavg_raw_noauto_3d] = pickle.load(f)

    # fit models
    data = tself_raw_noauto_3d
    K = 3
    random.seed(352021)
    # group
    mdl_grp = tm.TransitionModel(
        data, K, 'self', 'raw','noauto','grp',stickbreak = False
    )
    mdl_grp.fit()
    # dimensional
    mdl_dim = tm.TransitionModel(
        data, K, 'self', 'raw','noauto','dim',stickbreak = False
    )
    mdl_dim.fit()

    # make sparsemodel objects for inference
    sparse_grp = tm.SparseModel(
        data, K, 'self', 'raw','noauto','grp',stickbreak = False
    )
    sparse_dim = tm.SparseModel(
        data, K, 'self', 'raw','noauto','dim',stickbreak = False
    )

    # format single-feature artifical inputs for sparse inference
    a = [(torch.zeros(data.shape[1:3],dtype = torch.float64) + i) for i in np.arange(.1,1,.1)]
    obsmat = torch.stack(a, axis = 0)
    # coopt test_data learners to learn sparse artificial input
    sparse_grp.group_classify(mdl_grp,obsmat)
    sparse_dim.dimension_learn(mdl_dim, obsmat)

    # make sparse inferences
    hdist, hmeans, sdist, smeans = sparse_grp.group_infer(mdl_grp)
    ddist, dmeans = sparse_dim.dimension_infer(mdl_dim)

    # massage objects tobe r-friendly
    # def torch2np(mat):
    #     enu = np.ndenumerate(mat)
    #     for i in enu:
    #         ind = i[0]
    #         val = i[1]
    #         mat[ind] = val.detach().numpy()
    # hmeans = torch2np(hmeans)
    # smeans = torch2np(smeans)
    # dmeans = torch2np(dmeans)
    # other relevant objects for saving
    stor_grp = getattr(sparse_grp,'stor_grp')
    stor_grp_prb = getattr(sparse_grp,'stor_grp_prb')
    stor_dim_prb = getattr(sparse_dim,'stor_dim_prb')
    grp_map = getattr(mdl_grp,'map_estimates')
    dim_map = getattr(mdl_dim,'map_estimates')
    # save all relevant files for visualization
    with open('tomtom_final_raw_noauto_data4viz.pkl','wb') as f:
        pickle.dump([hmeans,smeans,dmeans,stor_grp,stor_grp_prb,stor_dim_prb,grp_map,dim_map],f)
