{
    "connectome":{
        "data_dir": "../../data",
        "name": "MICrONS"

    },
    "save_path":"../../data",
    "analyses":{
        "simplex_counts":{
            "save_suffix":"",
            "kwargs":{"threads": 8},
            "controls":{
                "seeds":[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                "types":{
                    "configuration_model":{},
                    "ER_shuffle":{},
                    "underlying_model":{},
                    "bishuffled_model":{},
                    "run_DD2":{
                        "kwargs":{
                            "a": 0.06705720999872188,
                            "b": 1.183000158541148e-05,
                            "xyz_labels":["x_nm", "y_nm", "z_nm"]
                        }
                    }
                }
            }
        },
        "node_degree":{
            "save_suffix":"",
            "kwargs":{"threads": 8},
            "controls":{
                "seeds":[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                "types":{
                    "configuration_model":{},
                    "ER_shuffle":{},
                    "underlying_model":{},
                    "bishuffled_model":{},
                    "run_DD2":{
                        "kwargs":{
                            "a": 0.06705720999872188,
                            "b": 1.183000158541148e-05,
                            "xyz_labels":["x_nm", "y_nm", "z_nm"]
                        }
                    }
                }
            }
        },
        "count_rc_edges_skeleta":{
            "save_suffix":"",
            "kwargs":{"threads": 10},
            "controls":{
                "seeds":[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                "types":{
                    "underlying_model":{},
                    "bishuffled_model":{}
                }
            }
        }
    }
}