import torch
from project_bisection import satisfies
import networkx as nx
from models.GDSS.utils.mol_utils import mols_to_nx, smiles_to_mols

def filtermap_constrained_graphs (test_graph_list, configt, constr_config=None):
    if constr_config is None:
        return test_graph_list
    adjs = torch.zeros(len(test_graph_list), configt.data.max_node_num, configt.data.max_node_num)
    for i, G in enumerate(test_graph_list):
        nG = G.number_of_nodes()
        adjs[i, :nG, :nG] = torch.tensor(nx.adjacency_matrix(G).todense())
    xs = torch.zeros (len(test_graph_list), configt.data.max_node_num, configt.data.max_feat_num)
    constr_val = satisfies(xs, adjs, constr_config).detach().cpu().numpy()
    return constr_val

def filtermap_constrained_smiles (test_smiles, configt, constr_config=None):
    if constr_config is None:
        return test_smiles
    gen_mols = smiles_to_mols (test_smiles)
    gen_graph_list = mols_to_nx (gen_mols)
    adjs = torch.zeros(len(gen_graph_list), configt.data.max_node_num, configt.data.max_node_num)
    for i, G in enumerate(gen_graph_list):
        nG = G.number_of_nodes()
        adjs[i, :nG, :nG] = torch.tensor(nx.adjacency_matrix(G, weight='label').todense())
    xs = torch.zeros (len(gen_graph_list), configt.data.max_node_num, configt.data.max_feat_num)
    atom_id_map = {'C': 0, 'N': 1, 'O': 2, 'F': 3, 'P': 4, 'S': 5, 'Cl': 6, 'Br': 7, 'I': 8}
    for i, G in enumerate(gen_graph_list):
        xs[i, torch.arange(len(G.nodes)), [atom_id_map[x['label']] for x in G.nodes().values()]] = 1
    constr_val = satisfies(xs, adjs, constr_config).detach().cpu().numpy()
    return constr_val
