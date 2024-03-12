import os
import pandas as pd
import h5py
import numpy as np
dir_path = './topological_sampling/working_dir/data/other/classifier/Parameter'
names = os.listdir(dir_path)

def get_scores(address):
    with h5py.File(address) as f:
        s = list(f['scores'])
    return s

#results = [[get_scores(dir_path+'/'+name+'/'+str(i)+'/results_components.h5') for i in range(50)] for name in names]
results = [get_scores(dir_path+'/'+name+'/merged/results_components.h5') for name in names]

D = pd.DataFrame(results)
D['parameter'] = names
D = D.set_index('parameter')
D.to_pickle('classification_results.pkl')

#results_mean = [[np.mean(j) for j in i] for i in results]
#results_tot_mean = [np.mean(j) for j in results_mean]
#D_mean = pd.DataFrame(results_mean)
#D_mean['overall_mean']=results_tot_mean
#D_mean['parameter']=names
#D_mean=D_mean.set_index('parameter')
#D_mean.to_pickle('classification_results_mean.pkl')
