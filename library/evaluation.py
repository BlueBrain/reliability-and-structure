# SPDX-FileCopyrightText: 2024 Blue Brain Project / EPFL
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# Function for comparing per-cell reliabilities using ROC analysis 

# Basic functions 
def renormalize_samples(sampleA,sampleB):
    "Renormalize the two samples together to lie between 0 and 1"
    t_min=np.amin([sampleA, sampleB])    
    t_max=np.amax([sampleA, sampleB])
    return (sampleA-t_min)/(t_max-t_min), (sampleB-t_min)/(t_max-t_min)

def get_ROC(a,b):
    '''a,b samples from two distribitions renormalized to lie between 0 and 1.
    Null hypothesis: for a threshold t, x>t is in sample a
    returns: 
    thresholds: all possible values in a \cup b
    FPR: False positive rate at all thresholds
    TPR: True prositive rate at all thresholds'''
    TP=[];FP=[]
    thresholds=np.unique(np.union1d(a,b))[::-1]
    for t in thresholds:
        TP.append(np.count_nonzero(a>t))
        FP.append(np.count_nonzero(b>t))
    TPR=np.array(TP)/a.size
    FPR=np.array(FP)/b.size
    return thresholds, FPR, TPR 

def auc(x,y):
    '''Compute area under the curve relative to the x=y axis'''
    return np.sum((y[1::]-x[1::])*(np.append(x,1)[2::]-x[1::]))

# Comparing across manipuations