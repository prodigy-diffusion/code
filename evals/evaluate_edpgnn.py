from models.GraphScoreMatching.evaluation.stats import eval_graph_list
from models.GraphScoreMatching.utils.mol_utils import load_smiles, canonicalize_smiles, mols_to_nx, smiles_to_mols
from models.GraphScoreMatching.utils.arg_helper import load_data
from models.GraphScoreMatching.evaluation.mmd import gaussian, gaussian_emd

from project_bisection import satisfies
from evals.filter_constr import filtermap_constrained_graphs, filtermap_constrained_smiles

from moses.metrics.metrics import get_all_metrics
import networkx as nx
import os
import torch
import pickle

TOTAL_MOLS = 10000
MIN_CONSTR_P = 0

mmd_kernels = {'degree':gaussian_emd, 
            'cluster':gaussian_emd, 
            'orbit':gaussian,
            'spectral':gaussian_emd}

mmd_methods = ['degree', 'cluster', 'orbit', 'spectral']

class DotDict(dict):
    """
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value

def evaluate (gen_graph_list, config, constr_config):
    _, test_graph_list = load_data(config, get_graph_list=True)

    # methods, kernels = load_eval_settings(config.dataset.data)
    test_constr = filtermap_constrained_graphs (test_graph_list, config, constr_config=constr_config)
    if test_constr.sum() < MIN_CONSTR_P * len(test_constr)/100:
        return {}
    test_graph_list = [graph for constr, graph in zip(test_constr, test_graph_list) if constr]
    result_dict = eval_graph_list(test_graph_list, gen_graph_list, methods=mmd_methods, kernels=mmd_kernels)
    adjs = torch.zeros(len(gen_graph_list), config.dataset.max_node_num, config.dataset.max_node_num)
    for i, G in enumerate(gen_graph_list):
        nG = G.number_of_nodes()
        adjs[i, :nG, :nG] = torch.tensor(nx.adjacency_matrix(G).todense())
    xs = torch.zeros (len(gen_graph_list), config.dataset.max_node_num, config.dataset.max_feat_num)
    constr_val = satisfies(xs, adjs, constr_config).sum().item()/len(adjs)
    result_dict['constr_val'] = constr_val
    return result_dict

def evaluate_mol (gen_smiles, config, constr_config):
    try:
        train_smiles, test_smiles = load_smiles(config.dataset.name, file_ext='_can')
    except:
        train_smiles, test_smiles = load_smiles(config.dataset.name)
        train_smiles, test_smiles = canonicalize_smiles(train_smiles), canonicalize_smiles(test_smiles)
    
    gen_mols = smiles_to_mols (gen_smiles)
    num_mols = len(gen_mols)
    gen_graph_list = mols_to_nx (gen_mols)
    
    # metrics
    with open(f'data/{config.dataset.name.lower()}_test_nx.pkl', 'rb') as f:
        test_graph_list = pickle.load(f)

    test_constr = filtermap_constrained_smiles (test_smiles, config, constr_config=constr_config)
    # print (test_constr.sum(), MIN_CONSTR_P * len(test_constr)/100)
    if test_constr.sum() < MIN_CONSTR_P * len(test_constr)/100:
        return {}
    test_smiles = [smiles for constr, smiles in zip(test_constr, test_smiles) if constr]
    test_graph_list = [graph for constr, graph in zip(test_constr, test_graph_list) if constr]
    
    scores = get_all_metrics(gen=gen_smiles, k=len(gen_smiles), device='cpu', n_jobs=8, 
                             test=test_smiles, train=train_smiles)
    # scores_nspdk = eval_graph_list(test_graph_list, gen_graph_list, methods=['nspdk'])['nspdk']
    result_dict = {}
    metrics = ['valid', f'unique@{len(gen_smiles)}', 'FCD/Test', 'Novelty']
    metric_names = ['valid', 'unique', 'fcd', 'novelty']
    for metric_name, metric in zip(metric_names, metrics):
        if metric_name == 'valid':
            result_dict[metric_name] = scores[metric] * num_mols / TOTAL_MOLS
        result_dict[metric_name] = scores[metric]
    # result_dict['nspdk'] = scores_nspdk
    result_dict['num_mols'] = num_mols
    adjs = torch.zeros(len(gen_graph_list), config.dataset.max_node_num, config.dataset.max_node_num)
    for i, G in enumerate(gen_graph_list):
        nG = G.number_of_nodes()
        adjs[i, :nG, :nG] = torch.tensor(nx.adjacency_matrix(G, weight='label').todense())
    xs = torch.zeros (num_mols, config.dataset.max_node_num, config.dataset.max_feat_num)
    atom_id_map = {'C': 0, 'N': 1, 'O': 2, 'F': 3, 'P': 4, 'S': 5, 'Cl': 6, 'Br': 7, 'I': 8}
    for i, G in enumerate(gen_graph_list):
        xs[i, torch.arange(len(G.nodes)), [atom_id_map[x['label']] for x in G.nodes().values()]] = 1
    constr_val = satisfies(xs, adjs, constr_config).sum().item()/len(adjs)
    result_dict['constr_val'] = constr_val
    return result_dict
