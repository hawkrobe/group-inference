

##### import packages
#base
import os
from collections import defaultdict
import numpy as np
import scipy.stats
from matplotlib import pyplot as plt
import random
import pyreadr
import numpy as np
import pandas as pd
import seaborn as sns
# %matplotlib inline
# %autosave 30

#pyro contingency
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.autoguide import AutoDelta
from pyro.optim import Adam
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete, Predictive
from pyro.ops.indexing import Vindex
from pyro.infer import MCMC, NUTS
import torch
from torch.distributions import constraints
pyro.enable_validation(True)

# import umap
# import plotly
# import plotly.graph_objs as go

#misc
import pickle
import torch.nn.functional as F
import itertools
import time

import arviz as az

# import homebrew modules
import tomtom_models as tm
import tomtom_util as tu

########## import data
# import pickled data
with open('tomtom_data_preprocessed.pkl','rb') as f:
    [tself_norm_all_3d, tself_norm_noauto_3d, tself_raw_all_3d, tself_raw_noauto_3d,
    ttarg_norm_all_3d, ttarg_norm_noauto_3d, ttarg_raw_all_3d, ttarg_raw_noauto_3d,
    tavg_norm_all_3d, tavg_norm_noauto_3d, tavg_raw_all_3d, tavg_raw_noauto_3d] = pickle.load(f)

tm.mtype = 'group'
tm.target = 'self' # 'self','targ','avg'
tm.dtype = 'norm' # 'norm','raw'
tm.auto = 'all' # 'noauto','all'
tm.stickbreak = False
tm.optim = pyro.optim.Adam({'lr': 0.0005, 'betas': [0.8, 0.99]})
tm.elbo = TraceEnum_ELBO(max_plate_nesting=1)

tm.K = 3

pyro.clear_param_store()
pyro.set_rng_seed(99)

# #declare dataset to be modeled
# dtname = 't{}_{}_{}_3d'.format(target, dtype, auto)
# print("running MCMC with: {}".format(dtname))
# data = globals()[dtname]

nuts_kernel = NUTS(tm.model)

mcmc = MCMC(nuts_kernel, num_samples=5000, warmup_steps=1000)
mcmc.run(tself_norm_all_3d)

posterior_samples = mcmc.get_samples()

abc = az.from_pyro(mcmc, log_likelihood=True)
az.stats.waic(abc.posterior.weights)
