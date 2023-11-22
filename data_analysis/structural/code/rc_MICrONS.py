from connalysis.network import topology
from connalysis import randomization 
import conntility
import warnings
import sys
sys.path.append('../../../library')
from read_connectomes import *
from rc_neighbors import *


# Load connectome
data_dir='/gpfs/bbp.cscs.ch/home/egassant/reliability_and_structure/data_analysis/data'# Directory where connectivity data is stored
exc_types=['23P', '4P', '5P_IT', '5P_NP', '5P_PT', '6CT', '6IT', 'BPC']
connectome=load_microns(data_dir).index("cell_type").isin(exc_types)
# Restricting to the boolean adjancency matrix for the analysis and checking the diagonal  
adj=connectome.matrix.astype('bool').tocsr() #remove weights of synapses
print(f"The connectome has {adj.diagonal().sum()} out of {adj.sum()} in the diagonal; these will be ignored.")
adj.setdiag(0)
adj.eliminate_zeros()
N=adj.shape[0]

# Metrics to be computed on neighbhoods
func_config={
    'nbd_size':{'function': lambda x : x.shape[0],
            'kwargs': {}},

    'edges':{'function': lambda x : x.sum(),
            'kwargs': {}},
    
    'degree':{'function':topology.node_degree,
            'kwargs': {'direction':("IN","OUT")}},

    'simplex_counts':{'function': topology.simplex_counts,
            'kwargs': {'threads':10}},
    'rc_densities':{'function':rc_densities,
            'kwargs': {}}}

# Configuration of analysis (includes properties specific to the connectome studied)
analysis_config={'graph': adj,#connectome.matrix.astype(bool).tocsr(),#[np.ix_(range(100),range(100))],
                 'out_file_prefix' : 'props_MICrONS',
                 'seed_CM': 0,
                 'func_config': func_config, # Dictionary of network properties to be computed 
                 # Directions of degrees to consider valid directions are a sublist of  "total", "IN", "OUT"
                 'degree_directions': ["total", "IN", "OUT"], 
                 # Distances between simplex counts considered
                 'sc_distances':{'euclidean_nodes': lambda x: norm_euclid(x, 'nodes'),
                                 'euclidean_edges': lambda x: norm_euclid(x,'edges'),
                                 'cosine':cosine},
                 'controls':{
                     'distance':{
                     'function':randomization.run_DD2,
                     # Parameters of distance dependent model, need to be precomputed separately.
                     # Adding parameters for MICrONS
                     'kwargs':{'n':N, 
                               'a':0.06705720999872188, 
                               'b':1.183000158541148e-05, 
                               'xyz':connectome.vertices[['x_nm', 'y_nm', 'z_nm']].to_numpy(),
                               'threads':8,
                               'seed':(10, 10)}},
                     'ER':{
                     'function':randomization.run_ER,
                     # Parameters of distance dependent model, need to be precomputed separately.
                     # Adding parameters for BBP E-E connectivity
                     'kwargs':{'n':N, 
                               'p':adj.sum()/((N)*(N-1)), 
                               'threads':10,
                               'seed':(10, 10)}}
                     },
                 # Path where to save the data
                'out_file_dir':'/gpfs/bbp.cscs.ch/home/egassant/reliability_and_structure/data_analysis/data/'}

# Compute properties 
warnings.filterwarnings('ignore') # Many neighborhoods are small and throw many warnings on distance metrics
compute_all_props(analysis_config, force_recompute=True)