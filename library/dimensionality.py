import numpy as np
import pandas as pd
from tqdm import tqdm
from connalysis.network import local 
import conntility

def correlation(x, mean_center=True):
    """Correlation between rows of matrix
    potentially faster than `1 - squareform(pdist(x, metrix="correlation"))`
    This speed difrence seems to depend on the size of the matrix"""
    if mean_center:
        x -= np.mean(x, axis=1).reshape(-1, 1)  # mean center trials
    x_norm = x / np.linalg.norm(x, axis=-1)[:, np.newaxis]
    return np.dot(x_norm, x_norm.T)

def corr_non_firing(corr,spikes, not_firing_corr=False):
    '''Correct nans of correlation matrix in one of two ways. 
    If not_firing_corr == True, the correlation between non-firing cells is set to 0. 
    Otherwise, it is set to 1'''
    corr=np.nan_to_num(corr)
    ind_0=np.where(np.abs(spikes).sum(axis=1)==0)[0] #indices of non-spiking neurons
    if not_firing_corr==False:
        corr[np.ix_(ind_0,ind_0)]=0
    elif not_firing_corr==True:
         corr[np.ix_(ind_0,ind_0)]=1
    return corr
def compute_ss_nbd(corr,nbd_indx):
    '''Compute spectrum of correlation submatrix of a neighborhood'''
    corr_nbd=corr[np.ix_(nbd_indx, nbd_indx)]
    return np.linalg.svd(corr_nbd-corr_nbd.mean(), compute_uv=False)
def get_dimensions(s,thresh=0.9):
    '''Return dimensions corresponding to s of: activity space, active tribe, tribe
    when thresholded at thresh'''
    normalized_cum_s=(np.cumsum(s)/s.sum())
    activity_dim=np.where(normalized_cum_s>thresh)[0][0]
    active_ts=np.where(np.isclose(normalized_cum_s,1))[0][0]
    ts=normalized_cum_s.size
    return activity_dim, active_ts, ts

def get_dimensions_nbds(connectome, corrs, spikes, all_nodes, centers, not_firing_corr=False, thresh=0.9):
    '''Compute the dimension of the correlation submatrix of the neighborhoods 
    with center in centers or for all nodes is all_nodes is True. 
    not_firing_corr=False resolves nans in correlation matrix to 0 of cells are not spiking, 
    otherwise it resolves them to 1'''
                        
    # Replace nan values with 0 for non firing cells
    corrs=corr_non_firing(corrs, spikes, not_firing_corr=not_firing_corr)
    # Get neighborhoods
    M=connectome.matrix.astype("bool")
    neighborhoods=local.neighborhood_indices(M, pre=True, post=True,all_nodes=all_nodes,centers=centers) 
    # Loop through neighborhoods 
    dim_df=pd.DataFrame(index=neighborhoods.index, columns=["actitivy_dimension", "active_ts", "ts"])
    for i in tqdm(range(len(neighborhoods))):
        nbd_idx=np.append(neighborhoods.index[i], neighborhoods.iloc[i])
        try: # This is required because sometimes SVD doesn't converge
            s=compute_ss_nbd(corrs, nbd_idx)
            dims=get_dimensions(s,thresh=thresh)
            assert dims[-1]==nbd_idx.shape[0], "There is an mismatch between tribe size and sub correlation matrix"
            dim_df.loc[nbd_idx[0]]=dims
        except: 
            print(f"SVD failed for node {neighborhoods.index[i]} so I'm skipping it")
    return dim_df

def get_spectrum_nbds(connectome, corrs, spikes, centers, all_nodes=False, not_firing_corr=False):
    '''Compute the spectrum of the correlation submatrix of the neighborhoods 
    with center in centers.  Not recomended for all nodes. 
    not_firing_corr=False resolves nans in correlation matrix to 0 of cells are not spiking, 
    otherwise it resolves them to 1'''
                        
    # Replace nan values with 0 for non firing cells
    corrs=corr_non_firing(corrs, spikes, not_firing_corr=not_firing_corr)
    # Get neighborhoods
    M=connectome.matrix.astype("bool")
    neighborhoods=local.neighborhood_indices(M, pre=True, post=True,all_nodes=all_nodes,centers=centers) 
    # Loop through neighborhoods 
    ss_df={center:None for center in neighborhoods.index}
    for i in tqdm(range(len(neighborhoods))):
        nbd_idx=np.append(neighborhoods.index[i], neighborhoods.iloc[i])
        try:
            s=compute_ss_nbd(corrs, nbd_idx)
            ss_df[nbd_idx[0]]=s
        except:
            print(f"SVD failed for node {neighborhoods.index[i]} so I'm skipping it")
    return ss_df