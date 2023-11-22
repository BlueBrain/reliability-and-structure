from connalysis.network import topology
from connalysis import randomization 
import scipy.sparse as sp
import conntility
import warnings
import sys
sys.path.append('../../../library')
from read_connectomes import *
from rc_neighbors import *


# Load connectome
# TODO: Update this with a zenodo release
mat_root='/gpfs/bbp.cscs.ch/data/scratch/proj9/bisimplices/simulations/BlobStimReliability_O1v5-SONATA_Baseline/working_dir/'
connectome=conntility.ConnectivityMatrix(sp.load_npz(f'{mat_root}connectivity.npz'), 
                                         vertex_properties=pd.read_pickle(f'{mat_root}neuron_info.pickle'))
connectome=connectome.index("synapse_class").isin("EXC")
adj=connectome.matrix.astype('bool').tocsr()
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
analysis_config={'graph': adj,
                 'out_file_prefix' : 'props_BBP',
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
                     # Adding parameters for BBP E-E connectivity
                     'kwargs':{'n':N, 
                               'a':0.11094128168168445, 
                               'b':0.007052547140545248,
                               'xyz':connectome.vertices[['x', 'y', 'z']].to_numpy(),
                               'threads':10,
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
compute_all_props(analysis_config, force_recompute=False)