# SPDX-FileCopyrightText: 2024 Blue Brain Project / EPFL
#
# SPDX-License-Identifier: AGPL-3.0-or-later

##
## Load packages
##

print('Loading packages',flush=True)
import sys
import os.path
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import load_npz
import scipy.sparse
import scipy.linalg
import pyflagser as pf
import pyflagsercount as pfc

##
## Set directories
##

root_dir = 'classification/network_based/'

##
## Load circuit
##

print('Loading circuit',flush=True)
adj_full = load_npz(root_dir+'TriDy/data/mc2.npz').toarray().astype(int)
exc_flag = np.load(root_dir+'exc_flag.npy')
adj = adj_full[np.ix_(exc_flag,exc_flag)]
adjt = np.transpose(adj)
nnum = adj.shape[0]
assert nnum == 26567, f"Excitatory-excitatory adjacency matrix expected with size 26567, instead found {nnum}"

##
## Load functions
##

print('Loading TriDy functions',flush=True)

def nx_to_np(directed_graph):
#  In: networkx directed graph
# Out: numpy array
	return nx.to_numpy_array(directed_graph,dtype=int)

def np_to_nx(adjacency_matrix):
#  In: numpy array
# Out: networkx directed graph
	return nx.from_numpy_array(adjacency_matrix,create_using=nx.DiGraph)

def open_neighbourhood(v, matrix=adj):
#  In: index
# Out: list of neighbours
	neighbours = np.unique(np.concatenate((np.nonzero(matrix[v])[0],np.nonzero(np.transpose(matrix)[v])[0])))
	neighbours.sort(kind='mergesort')
	return neighbours

def closed_neighbourhood(v, matrix=adj):
#  In: index
# Out: list of neighbours
	neighbours = np.unique(np.concatenate((np.nonzero(matrix[v])[0],np.nonzero(np.transpose(matrix)[v])[0])))
	neighbours.sort(kind='mergesort')
	return np.concatenate((np.array([v]),neighbours))

def open_tribe(v, matrix=adj):
#  In: index
# Out: adjacency matrix
	nhbd = open_neighbourhood(v)
	return matrix[np.ix_(nhbd,nhbd)]

def closed_tribe(v, matrix=adj):
#  In: index
# Out: adjacency matrix
	nhbd = closed_neighbourhood(v)
	return matrix[np.ix_(nhbd,nhbd)]

def spectral_gap(matrix, thresh=10, param='low'):
#  In: matrix
# Out: float
	current_spectrum = spectrum_make(matrix)
	current_spectrum = spectrum_trim_and_sort(current_spectrum, threshold_decimal=thresh)
	return spectrum_param(current_spectrum, parameter=param)

def spectrum_make(matrix):
#  In: matrix
# Out: list of complex floats
	assert np.any(matrix) , 'Error (eigenvalues): matrix is empty'
	eigenvalues = scipy.linalg.eigvals(matrix)
	return eigenvalues

def spectrum_trim_and_sort(spectrum, modulus=True, threshold_decimal=8):
#  In: list of complex floats
# Out: list of unique (real or complex) floats, sorted by modulus
	if modulus:
		return np.sort(np.unique(abs(spectrum).round(decimals=threshold_decimal)))
	else:
		return np.sort(np.unique(spectrum.round(decimals=threshold_decimal)))

def spectrum_param(spectrum, parameter):
#  In: list of complex floats
# Out: float
	assert len(spectrum) != 0 , 'Error (eigenvalues): no eigenvalues (spectrum is empty)'
	if parameter == 'low':
		if spectrum[0]:
			return spectrum[0]
		else:
			assert len(spectrum) > 1 , 'Error (low spectral gap): spectrum has only zeros, cannot return nonzero eigval'
			return spectrum[1]
	elif parameter == 'high':
		assert len(spectrum) > 1 , 'Error (high spectral gap): spectrum has one eigval, cannot return difference of top two'
		return spectrum[-1]-spectrum[-2]
	elif parameter == 'radius':
		return spectrum[-1]


def bls_matrix(matrix, reverse_flow=False):
#  in: tribe matrixrc
# out: bauer laplacian matrix
	current_size = len(matrix)
	return np.subtract(np.eye(current_size, dtype='float64'), tps_matrix(matrix, out_deg=reverse_flow))

def cls_matrix(matrix, is_strongly_conn=False):
#  in: numpy array
# out: numpy array
	matrix_nx = np_to_nx(matrix)
	return cls_matrix_fromdigraph(matrix_nx, matrix=matrix, matrix_given=True, is_strongly_conn=is_strongly_conn)

def cls_matrix_fromdigraph(digraph, matrix=np.array([]), matrix_given=False, is_strongly_conn=False):
#  in: networkx digraph
# out: numpy array
	digraph_sc = digraph
	matrix_sc = matrix
	# Make sure is strongly connected
	if not is_strongly_conn:
		largest_comp = max(nx.strongly_connected_components(digraph), key=len)
		digraph_sc = digraph.subgraph(largest_comp)
		matrix_sc = nx_to_np(digraph_sc)
	elif not matrix_given:
		matrix_sc = nx_to_np(digraph_sc)
	# Degeneracy: scc has size 1
	if not np.any(matrix_sc):
		return np.array([[0]])
	# Degeneracy: scc has size 2
	elif np.array_equal(matrix_sc,np.array([[0,1],[1,0]],dtype=int)):
		return np.array([[1,-0.5],[-0.5,1]])
	# No degeneracy
	else:
		return nx.directed_laplacian_matrix(digraph_sc)

def tps_matrix(matrix, out_deg=True):
#  in: tribe matrix
# out: transition probability matrix
	current_size = len(matrix)
	if out_deg:
		degree_vector = [np.count_nonzero(matrix[i]) for i in range(current_size)]
	else:
		matrixt = np.transpose(matrix)
		degree_vector = [np.count_nonzero(matrixt[i]) for i in range(current_size)]
	inverted_degree_vector = [0 if not d else 1/d for d in degree_vector]
	return np.matmul(np.diagflat(inverted_degree_vector),matrix)

def tcc_adjacency(matrix, chief_containment):
    outdeg = np.count_nonzero(matrix[0])
    indeg = np.count_nonzero(np.transpose(matrix)[0])
    repdeg = np.count_nonzero(np.multiply(matrix[0],np.transpose(matrix)[0]))
    totdeg = indeg+outdeg
    numerator = 0 if len(chief_containment) < 3 else chief_containment[2]
    denominator = (totdeg*(totdeg-1)-(indeg*outdeg+repdeg))
    if denominator == 0:
        return 0
    return numerator/denominator

def ccc_adjacency(matrix):
    deg = np.count_nonzero(matrix[0])+np.count_nonzero(np.transpose(matrix)[0])
    repdeg = np.count_nonzero(np.multiply(matrix[0],np.transpose(matrix)[0]))
    numerator = np.linalg.matrix_power(matrix+np.transpose(matrix),3)[0][0]
    denominator = 2*(deg*(deg-1)-2*repdeg)
    if denominator == 0:
        return 0
    return numerator/denominator

def dc_adjacency(matrix, chief_containment, coeff_index=2):
#  in: tribe matrix
# out: float
	assert coeff_index >= 2, 'Assertion error: Density coefficient must be at least 2'
	if len(chief_containment) <= coeff_index:
		density_coeff = 0
	elif chief_containment[coeff_index] == 0:
		density_coeff = 0
	else:
		numerator = coeff_index*chief_containment[coeff_index]
		denominator = (coeff_index+1)*(len(matrix)-coeff_index)*chief_containment[coeff_index-1]
		if denominator == 0:
			density_coeff = 0
		else:
			density_coeff = numerator/denominator
	return density_coeff

def nbc_adjacency(matrix, chief_index=0):
#  in: tribe matrix
# out: float
	flagser_output = pf.flagser.flagser_unweighted(matrix,directed=True)
	cells = flagser_output['cell_count']
	bettis = flagser_output['betti']
	while (cells[-1] == 0) and (len(cells) > 1):
		cells = cells[:-1]
	while (bettis[-1] == 0) and (len(bettis) > 1):
		bettis = bettis[:-1]
	normalized_betti_list = [(i+1)*bettis[i]/cells[i] for i in range(min(len(bettis),len(cells)))]
	return sum(normalized_betti_list)

##
## Define the main function
##

def compute_parameter(job):
	
	# Set up dictionary of computed parameters to save
	save_dict = {}


	# (12 params) High, low, radius spectral gaps of four spectra
	if job == 'spectral':
		print('Computing 12 spectral gaps',flush=True)
		asg = []; asg_low = []; asg_radius = []
		blsg = []; blsg_low = []; blsg_radius = []
		clsg = []; clsg_high = []; clsg_radius = []
		tpsg = []; tpsg_low = []; tpsg_radius = []

		# Iterate
		for n in range(nnum):
			if n%100 == 0:
				print(str(n)+' ',end='',flush=True)
			tribe = closed_tribe(n)

			# Adjacency spectrum
			spec = spectrum_trim_and_sort(spectrum_make(tribe))
			asg.append(spectrum_param(spec,'high'))
			asg_low.append(spectrum_param(spec,'low'))
			asg_radius.append(spectrum_param(spec,'radius'))
			
			# Bauer Laplacian spectrum
			mat = bls_matrix(tribe)
			spec = spectrum_trim_and_sort(spectrum_make(mat))
			blsg.append(spectrum_param(spec,'high'))
			blsg_low.append(spectrum_param(spec,'low'))
			blsg_radius.append(spectrum_param(spec,'radius'))

			# Chung Laplacian spectrum
			mat = cls_matrix(tribe)
			spec = spectrum_trim_and_sort(spectrum_make(mat))
			clsg.append(spectrum_param(spec,'low'))
			clsg_high.append(spectrum_param(spec,'high'))
			clsg_radius.append(spectrum_param(spec,'radius'))

			# Transition probability spectrum
			mat = tps_matrix(tribe)
			spec = spectrum_trim_and_sort(spectrum_make(mat))
			tpsg.append(spectrum_param(spec,'low'))
			tpsg_low.append(spectrum_param(spec,'high'))
			tpsg_radius.append(spectrum_param(spec,'radius'))

		print('\nRecording 12 spectral gaps',flush=True)
		save_dict['asg'] = np.array(asg)
		save_dict['asg_low'] = np.array(asg_low)
		save_dict['asg_radius'] = np.array(asg_radius)
		save_dict['blsg'] = np.array(blsg)
		save_dict['blsg_low'] = np.array(blsg_low)
		save_dict['blsg_radius'] = np.array(blsg_radius)
		save_dict['clsg'] = np.array(clsg)
		save_dict['clsg_high'] = np.array(clsg_high)
		save_dict['clsg_radius'] = np.array(clsg_radius)
		save_dict['tpsg'] = np.array(tpsg)
		save_dict['tpsg_low'] = np.array(tpsg_low)
		save_dict['tpsg_radius'] = np.array(tpsg_radius)


	# (6 params) High, low, radius spectral gaps of two spectra that can be reversed
	elif job == 'spectral_reverse':
		print('Computing 6 spectral gaps',flush=True)
		blsg = []; blsg_low = []; blsg_radius = []
		tpsg = []; tpsg_low = []; tpsg_radius = []

		# Iterate
		for n in range(nnum):
			if n%100 == 0:
				print(str(n)+' ',end='',flush=True)
			tribe = closed_tribe(n)

			# Bauer Laplacian spectrum
			mat = bls_matrix(tribe,reverse_flow=True)
			spec = spectrum_trim_and_sort(spectrum_make(mat))
			blsg.append(spectrum_param(spec,'high'))
			blsg_low.append(spectrum_param(spec,'low'))
			blsg_radius.append(spectrum_param(spec,'radius'))

			# Transition probability spectrum
			mat = tps_matrix(tribe,out_deg=False)
			spec = spectrum_trim_and_sort(spectrum_make(mat))
			tpsg.append(spectrum_param(spec,'low'))
			tpsg_low.append(spectrum_param(spec,'high'))
			tpsg_radius.append(spectrum_param(spec,'radius'))

		print('\nRecording 6 spectral gaps',flush=True)
		save_dict['blsg_reversed'] = np.array(blsg)
		save_dict['blsg_reversed_low'] = np.array(blsg_low)
		save_dict['blsg_reversed_radius'] = np.array(blsg_radius)
		save_dict['tpsg_reversed'] = np.array(tpsg)
		save_dict['tpsg_reversed_low'] = np.array(tpsg_low)
		save_dict['tpsg_reversed_radius'] = np.array(tpsg_radius)

	# (17 params) How many simplices in tribe, how many simplices chief is in (up to dim 7)
	elif job == 'simplices':
		print('Computing 16 simplex counts and 1 EC count',flush=True)
		simp0 = []; simp1 = []; simp2 = []; simp3 = []; simp4 = []; simp5 = []; simp6 = []; simp7 = []
		simplists = [simp0, simp1, simp2, simp3, simp4, simp5, simp6, simp7]
		cont0 = []; cont1 = []; cont2 = []; cont3 = []; cont4 = []; cont5 = []; cont6 = []; cont7 = []
		contlists = [cont0, cont1, cont2, cont3, cont4, cont5, cont6, cont7]
		ec = []

		# Iterate
		for n in range(nnum):
			if n%100 == 0:
				print(str(n)+' ',end='',flush=True)

			# Compute
			ctribe = closed_tribe(n)
			otribe = open_tribe(n)
			cdict = pfc.flagser_count(ctribe)
			odict = pfc.flagser_count(otribe)

			# Record
			cdict_len = len(cdict['cell_counts'])
			odict_len = len(odict['cell_counts'])
			for dim in range(8): 
				if cdict_len > dim:
					simplists[dim].append(cdict['cell_counts'][dim])
					if odict_len > dim:
						contlists[dim].append(cdict['cell_counts'][dim]-odict['cell_counts'][dim])
					else:
						contlists[dim].append(cdict['cell_counts'][dim])
				else:
					simplists[dim].append(0)
					contlists[dim].append(0)
			ec.append(cdict['euler'])

		print('\nRecording 16 simplex counts and 1 EC count',flush=True)
		save_dict['0simplex'] = np.array(simp0)
		save_dict['1simplex'] = np.array(simp1)
		save_dict['2simplex'] = np.array(simp2)
		save_dict['3simplex'] = np.array(simp3)
		save_dict['4simplex'] = np.array(simp4)
		save_dict['5simplex'] = np.array(simp5)
		save_dict['6simplex'] = np.array(simp6)
		save_dict['7simplex'] = np.array(simp7)
		save_dict['0containment'] = np.array(cont0)
		save_dict['1containment'] = np.array(cont1)
		save_dict['2containment'] = np.array(cont2)
		save_dict['3containment'] = np.array(cont3)
		save_dict['4containment'] = np.array(cont4)
		save_dict['5containment'] = np.array(cont5)
		save_dict['6containment'] = np.array(cont6)
		save_dict['7containment'] = np.array(cont7)
		save_dict['ec'] = np.array(ec)


	# (3 params) Degrees
	elif job == 'degrees':
		print('Computing 3 degree counts',flush=True)
		out_degree = []; in_degree = []; degree = []

		# Iterate
		for n in range(nnum):
			if n%100 == 0:
				print(str(n)+' ',end='',flush=True)

			# Compute
			odeg = np.count_nonzero(adj[n])
			ideg = np.count_nonzero(adjt[n])

			# Record
			out_degree.append(odeg)
			in_degree.append(ideg)
			degree.append(odeg+ideg)

		print('\nRecording 3 degree counts',flush=True)
		save_dict['out_degree'] = np.array(out_degree)
		save_dict['in_degree'] = np.array(in_degree)
		save_dict['degree'] = np.array(degree)


	# (2 params) Clustering coefficients
	# Assumes job 'simplices' has been done
	elif job == 'cc':
		print('Computing 2 clustering coefficients',flush=True)
		tcc = []; fcc = []

		# Check containment count has been done
		files = np.array([os.path.isfile(root_dir+'parameters/'+str(i)+'containment.npy') for i in range(8)])
		if not np.all(files):
			print('Simplex containment jobs have not been done, exiting',flush=True)
			exit()
		dict_temp = {str(i)+'containment':np.load(root_dir+'parameters/'+str(i)+'containment.npy') for i in range(8)}
		df_temp = pd.DataFrame.from_dict(dict_temp)

		# Iterate
		for n in range(nnum):
			if n%100 == 0:
				print(str(n)+' ',end='',flush=True)

			# Compute 
			tribe = closed_tribe(n)
			current_tcc = tcc_adjacency(tribe,df_temp.iloc[n].values)
			current_fcc = ccc_adjacency(tribe) 

			# Record
			tcc.append(current_tcc)
			fcc.append(current_fcc)

		print('\nRecording 2 clustering coefficients',flush=True)
		save_dict['tcc'] = np.array(tcc)
		save_dict['fcc'] = np.array(fcc)


	# (5 params) Density coefficients
	# Assumes job 'simplices' has been done
	elif job == 'dc':
		print('Computing 5 density coefficients',flush=True)
		dc2 = []; dc3 = []; dc4 = []; dc5 = []; dc6 = []

		# Check containment count has been done
		files = np.array([os.path.isfile(root_dir+'parameters/'+str(i)+'containment.npy') for i in range(8)])
		if not np.all(files):
			print('Simplex containment jobs have not been done, exiting',flush=True)
			exit()
		dict_temp = {str(i)+'containment':np.load(root_dir+'parameters/'+str(i)+'containment.npy') for i in range(8)}
		df_temp = pd.DataFrame.from_dict(dict_temp)

		# Iterate
		for n in range(nnum):
			if n%100 == 0:
				print(str(n)+' ',end='',flush=True)

			# Compute 
			tribe = closed_tribe(n)
			current_containment = df_temp.iloc[n].values
			current_dc2 = dc_adjacency(tribe, current_containment, coeff_index = 2)
			current_dc3 = dc_adjacency(tribe, current_containment, coeff_index = 3)
			current_dc4 = dc_adjacency(tribe, current_containment, coeff_index = 4)
			current_dc5 = dc_adjacency(tribe, current_containment, coeff_index = 5)
			current_dc6 = dc_adjacency(tribe, current_containment, coeff_index = 6)

			# Record
			dc2.append(current_dc2)
			dc3.append(current_dc3)
			dc4.append(current_dc4)
			dc5.append(current_dc5)
			dc6.append(current_dc6)

		print('\nRecording 5 density coefficients',flush=True)
		save_dict['dc2'] = np.array(dc2)
		save_dict['dc3'] = np.array(dc3)
		save_dict['dc4'] = np.array(dc4)
		save_dict['dc5'] = np.array(dc5)
		save_dict['dc6'] = np.array(dc6)

	# (1 param) Normalized Betti coefficient
	elif job == 'nbc':
		print('Computing 1 normalized betti coefficient',flush=True)
		nbc = []

		# Iterate
		for n in range(nnum):
			if n%100 == 0:
				print(str(n)+' ',end='',flush=True)

			# Compute
			tribe = closed_tribe(n)
			current_nbc = nbc_adjacency(tribe)

			# Record
			nbc.append(current_nbc)

		print('\nRecording 1 normalized betti coefficient',flush=True)
		save_dict['nbc'] = np.array(nbc)


	# (2 params) Reciprocal connections
	elif job == 'rc':
		print('Computing 2 reciprocal connection coefficients',flush=True)
		rc = []; rcchief = []

		# Iterate
		for n in range(nnum):
			if n%100 == 0:
				print(str(n)+' ',end='',flush=True)

			# Compute
			tribe = closed_tribe(n)
			current_rc = np.count_nonzero(np.multiply(tribe,np.transpose(tribe)))//2
			current_rcchief = np.count_nonzero(np.multiply(tribe[0],np.transpose(tribe)[0]))

			# Record
			rc.append(current_rc)
			rcchief.append(current_rcchief)

		print('\nRecording 2 reciprocal connection coefficients',flush=True)
		save_dict['rc'] = np.array(rc)
		save_dict['rcchief'] = np.array(rcchief)


	# Quit if not recognized job name
	else:
		print('Requested job name \''+job+'\' is not one of recognized keywords. Exiting.',flush=True)
		exit()

	# Save computed parameters
	for key in save_dict.keys():
		print('Saving parameter '+key,flush=True)
		np.save(root_dir+'parameters/'+key+'.npy',save_dict[key])

	print('All done',flush=True)

##
## Read job to do
##

print('Reading job to do',flush=True)
job = sys.argv[1]

##
## Do the job
##

if __name__ == "__main__":
	compute_parameter(job)
