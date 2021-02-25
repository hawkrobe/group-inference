import itertools
import numpy as np
import pickle
import torch

nstim = 3

cellcombo = list(itertools.combinations(np.arange(60),3))
def parse_cellcombo_segment(ind):
    # map segments of cellcombo to index
    endpoint = np.ceil(len(cellcombo)/500)*500
    indmap = np.arange(0, endpoint, 500)
    startpoint = indmap[ind]
    if startpoint == 34000:
        seg = np.arange(startpoint, startpoint + 220)
    else:
        seg = np.arange(startpoint, startpoint + 500)
    return seg.astype(int)


nfile = np.ceil(len(cellcombo)/500)
stor_grp_3feat_all = []
stor_grp_prb_3feat_all = []
for i in np.arange(nfile):
    comb_segment = parse_cellcombo_segment(int(i))
    fname = 'stor_grp_3feat_chunk_{}to{}.pkl'.format(comb_segment[0],comb_segment[-1])
    with open(fname,'rb') as f:
        [p1,p2] = pickle.load(f)
        stor_grp_3feat_all.append(p1)
        stor_grp_prb_3feat_all.append(p2)
stor_grp_3feat_all = torch.cat(stor_grp_3feat_all)
stor_grp_prb_3feat_all = torch.cat(stor_grp_prb_3feat_all)

with open('stor_grp_3feat.pkl','wb') as f:
    pickle.dump([stor_grp_3feat_all,stor_grp_prb_3feat_all],f)
