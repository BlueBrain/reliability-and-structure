# SPDX-FileCopyrightText: 2024 Blue Brain Project / EPFL
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# Function to read several connectomes and generate from these a ConnectivityMatrix object using conntility
# The code is taken from:
# https://github.com/BlueBrain/ConnectomeUtilities/blob/main/examples

import conntility



def load_drosophila(data_dir):
    """Create a ConnectivityMatrix object of the Drosphila Larva brain  using conntility.  
    This code is copied from 
    https://github.com/BlueBrain/ConnectomeUtilities/blob/main/examples/Fly%20larva%20-%20a%20non-sonata-based%20example.ipynb
    
    The data used is referring to:
    Winding et al., 2023 - The connectome of an insect brain. Science: https://www.science.org/doi/10.1126/science.add9330
    
    To run this function, download the data in the following link and store it in a folder called data
    https://www.science.org/doi/suppl/10.1126/science.add9330/suppl_file/science.add9330_data_s1_to_s4.zip
    
    Read neuron info. Notably, in the raw data corresponding neurons in the left and right hemisphere are represented in the same row. 
    In this function they are split into separate rows."""

    import pandas as pd
    import numpy as np
    from scipy import sparse
    
    #Loading neuron information
    fn_neurons = f"{data_dir}/science.add9330_data_s1_to_s4/science.add9330_data_s2.csv"
    neurons_raw = pd.read_csv(fn_neurons)
    str_id_left = "left_id"; str_id_right = "right_id"
    neurons_left = neurons_raw.loc[neurons_raw[str_id_left] != "no pair"].rename(columns={str_id_left: "id"})
    neurons_left["hemisphere"] = "left"
    neurons_left = neurons_left[neurons_left.columns.drop(str_id_right)]
    neurons_right = neurons_raw.loc[neurons_raw[str_id_right] != "no pair"].rename(columns={str_id_right: "id"})
    neurons_right["hemisphere"] = "right"
    neurons_right = neurons_right[neurons_right.columns.drop(str_id_left)]
    neurons = pd.concat([neurons_left, neurons_right], axis=0)
    neurons["id"] = neurons["id"].apply(lambda _x: int(_x))
    neurons = neurons.sort_values("id").reset_index(drop=True)
    neuron_cols = neurons.columns.drop("id")

    #Functions to read connectivity matrices
    def fill_in_missing_neurons(neurons, dict_raw_mats):
        """Finds neuron skeleton ids that are used in the connectivity matrices that are not in the table of neuron properties 
        for whatever reason and fills in dummy data (mostly "unknown") for them."""
        
        default = {"celltype": "unknown", "additional_annotations": "unknown",
                   "level_7_cluster": "no cluster", "hemisphere": "unknown"}
        
        lst_raw_mats = list(dict_raw_mats.values())
        mis_ids = []
        for con_raw in lst_raw_mats:
            id_x = list(map(int, con_raw.columns.values))
            id_y = list(map(int, con_raw.index.values))
            assert(np.all(np.array(id_x) == np.array(id_y)))
    
            mis_ids.extend([_x for _x in id_x if _x not in neurons["id"].values])
        mis_ids = np.unique(mis_ids)
        mis_entries = [default.copy() for _x in mis_ids]
        [_x.update({"id": _id}) for _x, _id in zip(mis_entries, mis_ids)]
        mis_entries = pd.DataFrame.from_records(mis_entries)
    
        neurons = pd.concat([neurons, mis_entries], axis=0).sort_values("id").reset_index(drop=True)
        return neurons
    
    def to_edge_dataframe(neurons, dict_raw_mats):
        """Converts the matrices into a sparse representation, associates the connections with their types (axo-dendritic, etc.) 
        and concatenates the four connectomes. """
        neuron_bins = np.hstack([neurons["id"].values, neurons["id"].values[-1] + 1])
    
        res = []
        for tp_raw, con_raw in dict_raw_mats.items():
            id_x = list(map(int, con_raw.columns.values))
            id_y = list(map(int, con_raw.index.values))
            assert(np.all(np.array(id_x) == np.array(id_y)))
            assert np.in1d(id_x, neurons["id"]).all()
            idxx = np.digitize(id_x, bins=neuron_bins) - 1
            assert len(np.unique(idxx)) == len(idxx)
    
            tmp_M = sparse.coo_matrix(con_raw.values)
            df = pd.DataFrame({"row": idxx[tmp_M.row], "col": idxx[tmp_M.col],
                                   "count": tmp_M.data, "type": [tp_raw for _x in range(tmp_M.nnz)]})
            res.append(df)
        return pd.concat(res, axis=0).reset_index(drop=True)

    # Create ConnectivityMatrix object with the data loaded 
    dict_con_files = {"axo-dendritic": f"{data_dir}/science.add9330_data_s1_to_s4/Supplementary-Data-S1/ad_connectivity_matrix.csv",
                      "axo-axonic": f"{data_dir}/science.add9330_data_s1_to_s4/Supplementary-Data-S1/aa_connectivity_matrix.csv",
                      "dendro-dendritic": f"{data_dir}/science.add9330_data_s1_to_s4/Supplementary-Data-S1/dd_connectivity_matrix.csv",
                      "dendro-axonic": f"{data_dir}/science.add9330_data_s1_to_s4/Supplementary-Data-S1/da_connectivity_matrix.csv"}
    dict_con_data = dict([(_k, pd.read_csv(_v, index_col=0)) for _k, _v in dict_con_files.items()])
    neurons_augmented = fill_in_missing_neurons(neurons, dict_con_data)
    edges = to_edge_dataframe(neurons_augmented, dict_con_data)
    edge_indices = edges[["row", "col"]]
    edge_properties = edges[edges.columns.drop(["row", "col"])]
    M = conntility.ConnectivityMatrix(edge_indices, edge_properties=edge_properties,
                                      vertex_properties=neurons_augmented, 
                                      default_edge_property="count",
                                      shape=(len(neurons_augmented), len(neurons_augmented)))
    
    return M


def load_C_elegans_stages(data_dir):
    """Create a ConnectivityMatrix object of the eight developmenta stages of C. elegans  using conntility.  
    This code is copied from 
    https://github.com/BlueBrain/ConnectomeUtilities/blob/main/examples/C%20elegans%20-%20a%20non-sonata-based%20example.ipynb
    
    The data used is referring to:
    Witvliet et al: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8756380/ 
    Annotations and properties for the individual neurons are taken from wormatlas.org.
    
    To run this function, download the data in the following links and place them under data
    https://www.wormatlas.org/images/NeuronType.xls
    https://wormwiring.org/pages/witvliet.html 
    """
    import pandas as pd
    import numpy as np
    import glob 

    # Loading neuron data 
    fns = glob.glob(f"{data_dir}/witvliet_2020*.xlsx")
    stage = [int(_fn[len(data_dir)+15]) for _fn in fns]
    idx = np.argsort(stage)
    fns = [fns[_i] for _i in idx]
    stage = [stage[_i] for _i in idx]
    rn_dict = {"pre": "row", "post": "col"}
    syns = [pd.read_excel(_fn).set_index("type")
           for _fn in fns]
    neurons = pd.read_excel(f"{data_dir}/NeuronType.xls")
    syns_chem = pd.concat([syn.loc["chemical"] for syn in syns],
                              axis=0, names=["stage"], keys=stage).reset_index().rename(columns=rn_dict)
    syns_elec = pd.concat([syn.loc["electrical"] for syn in syns],
                              axis=0, names=["stage"], keys=stage).reset_index().rename(columns=rn_dict)
    # Separate the chemical and gap junction connectome and do some minor reformatting.
    syns_chem = syns_chem.pivot(index=["row", "col"], columns="stage", values="synapses").fillna(0).astype(int)
    syns_elec = syns_elec.pivot(index=["row", "col"], columns="stage", values="synapses").fillna(0).astype(int)
    
    syns_all = pd.concat([syns_chem, syns_elec], axis=0)
    idxx = syns_all.index.to_frame()
    
    used_neurons = idxx.reset_index(drop=True).unstack().drop_duplicates().values
    missing = np.setdiff1d(used_neurons, neurons["Neuron"].values)
    
    missing_df = pd.DataFrame.from_records([
        dict([(k, {"Neuron": name,
                   "Soma Region": "-",  # Placeholders for string-type properties
                   "Span": "-",
                   "AY Ganglion Designation": "-"}.get(k, np.NaN)) for k in neurons.columns])
        for name in missing
    ])
    # Some entries of the connectivity data are not represented in the DataFrame of neuron information. 
    # Thus, we have to create additional rows for the missing entries and fill them with mostly "NaN" and concatenate.
    
    syns_all = pd.concat([syns_chem, syns_elec], axis=0)
    idxx = syns_all.index.to_frame()
    used_neurons = idxx.reset_index(drop=True).unstack().drop_duplicates().values
    missing = np.setdiff1d(used_neurons, neurons["Neuron"].values)
    missing_df = pd.DataFrame.from_records([
        dict([(k, {"Neuron": name,
                   "Soma Region": "-",  # Placeholders for string-type properties
                   "Span": "-",
                   "AY Ganglion Designation": "-"}.get(k, np.NaN)) for k in neurons.columns])
        for name in missing
    ])
    neurons = pd.concat([neurons, missing_df], axis=0).reset_index(drop=True)
    # Look up for each connection the indices of participating nodes in the neuron information table and 
    # put the results into a separate DataFrame.
    nrn_idxx = neurons["Neuron"].reset_index().set_index("Neuron")["index"]
    edges_chem = syns_chem.index.to_frame().applymap(lambda x: nrn_idxx[x]).reset_index(drop=True)
    edges_elec = syns_elec.index.to_frame().applymap(lambda x: nrn_idxx[x]).reset_index(drop=True)
    # Create ConnectivityMatrices objects from 
    # The indices of the edges
    # The table of synapse counts at different stages for each edge (in the same order as input 1)
    # The table of vertex properties
    
    M_chem = conntility.ConnectivityMatrix(edges_chem, edge_properties=syns_chem.reset_index(drop=True),
                                           vertex_properties=neurons, shape=(len(neurons), len(neurons)))
    M_elec = conntility.ConnectivityMatrix(edges_elec, edge_properties=syns_elec.reset_index(drop=True),
                                           vertex_properties=neurons, shape=(len(neurons), len(neurons)))
    # Concatenate into a single DataFrame electrical and chemical synapses 
    edge_indices = pd.concat([
        M_chem._edge_indices,
        M_elec._edge_indices
    ], axis=0).reset_index(drop=True)
    
    _edge_prop_c = M_chem._edges.copy()
    _edge_prop_c["type"] = "chemical"
    _edge_prop_e = M_elec._edges.copy()
    _edge_prop_e["type"] = "electrical"
    _edge_prop = pd.concat([_edge_prop_c, _edge_prop_e], axis=0).reset_index(drop=True)
    
    M_all = conntility.ConnectivityMatrix(edge_indices, edge_properties=_edge_prop,
                                         vertex_properties=neurons, shape=(len(neurons), len(neurons)))
    print("Warning!!! When accessing the adjacency as a sparse matrix using the .matrix property:\n\
    Connections that are not present at a given stage, but at other stages will be represented as edges,\n\
    but with a value of ``0`` synapses associated with them.  For structural analysis always use .eliminate_zeros")
    
    return M_all

import conntility

def load_microns(data_dir, data_type="condensed", restrict_to_interior=True, 
                 interval_z = [700000, 1000000], interval_x = [650000, 950000]):
    """Load the connectivity of the of the IARPA MICrONS mm^3 dataset (https://www.microns-explorer.org/cortical-mm3), 
    formatted into a ConnectivityMatrix object as provided in https://zenodo.org/record/8364070
    To run this function it is required to dowload the data from zenodo and place it under data.
    The code of the function is extracted from the notebook ``Microns check edge effect.ipynb`` in the zenodo link above.
    """
    fn_mat = f"{data_dir}/microns_mm3_connectome.h5" 
    name_dset = "condensed" #"full" 
    M = conntility.ConnectivityMatrix.from_h5(fn_mat, name_dset)
    if restrict_to_interior:
        # Excluding boundary and restricting to exc types 
        M = M.index("x_nm").gt(interval_x[0]).index("x_nm").lt(interval_x[1]).index("z_nm").gt(interval_z[0]).index("z_nm").lt(interval_z[1])
    return M
