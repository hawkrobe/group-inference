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
import torch.multiprocessing as multiprocessing

from sklearn.model_selection import KFold

import tomtom_models_rewrite as tm

if __name__ == '__main__':
    import pickle
    import time
    import sys
    with open('tomtom_data_preprocessed_withadded.pkl','rb') as f:
        [tself_norm_all_3d, tself_norm_noauto_3d, tself_raw_all_3d, tself_raw_noauto_3d,
        ttarg_norm_all_3d, ttarg_norm_noauto_3d, ttarg_raw_all_3d, ttarg_raw_noauto_3d,
        tavg_norm_all_3d, tavg_norm_noauto_3d, tavg_raw_all_3d, tavg_raw_noauto_3d] = pickle.load(f)

    k = int(sys.argv[1])

    split_id = int(sys.argv[2])

    trunc_data = tself_raw_noauto_3d[:,:,:] # truncated data for quick debug

    phn = tm.ParallelEvaluator(trunc_data, 'self','raw','noauto','dim', maxk = 10, K = k, random_state = 888)
    q = phn.evaluate_parallel_alt(split_id) # pass split id for manual parallelization

    with open('kfold_metrics_dim_k{}_split{}.pkl'.format(k,split_id),'wb') as f:
        pickle.dump([q],f)
