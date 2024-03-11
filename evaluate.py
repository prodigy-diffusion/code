# import pickle
import pickle5 as pickle
import csv
import os
import argparse
import sys
from utils import get_new_log_name, get_new_log_folder_name, save_setting

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='constr')
    parser.add_argument('--model', type=str, default='GDSS')
    parser.add_argument('--dataset', type=str, default='community_small')
    parser.add_argument('--constraint', type=str, default='configs/none/constraint.yaml')
    parser.add_argument('--method', type=str, default='configs/none/method.yaml')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_known_args(sys.argv[1:])
    from utils import CONSTR_CONFIG, METHOD_CONFIG
    if args.model == 'GDSS':
        from models.GDSS.utils.logger import set_log
        from models.GDSS.utils.loader import load_ckpt
        from models.GDSS.parsers.config import get_config
        from evals.evaluate_gdss import evaluate, evaluate_mol
        config = get_config(f"sample_{args.dataset}", seed=args.seed)
        ckpt_dict = load_ckpt(config, 'cpu')
        configt = ckpt_dict['config']
        log_folder_name, log_dir, _ = set_log(configt, is_train=False)
        log_name = f"{config.ckpt}-sample"
        log_folder_name = get_new_log_folder_name(log_folder_name)
        log_dir = get_new_log_folder_name(log_dir)
        log_name = get_new_log_name(log_name)
        if args.dataset in ['QM9', 'ZINC250k']:
            if not os.path.exists(f"./GDSS/{log_dir}/{log_folder_name}/{log_name}.txt"):
                print (f"./GDSS/{log_dir}/{log_folder_name}/{log_name}.txt not found")
            else:
                gen_smiles = []
                with open (f"{log_dir}{log_name}.txt", 'r') as f:
                    for line in f:
                        gen_smiles.append(line[:-1])
                results_dict = evaluate_mol (gen_smiles, configt, config, CONSTR_CONFIG)
                print (results_dict)
        else:
            if not os.path.exists(f'./GDSS/samples/pkl/{log_folder_name}/{log_name}.pkl'):
                print (f'./GDSS/samples/pkl/{log_folder_name}/{log_name}.pkl not found')
            else:
                with open(f'./GDSS/samples/pkl/{log_folder_name}/{log_name}.pkl', 'rb') as f:
                    gen_graph_list = pickle.load(f)
                results_dict = evaluate (gen_graph_list, configt, config, CONSTR_CONFIG)
                print (results_dict)
    elif args.model == 'DruM':
        from models.DruM.DruM_2D.utils.logger import set_log
        from models.DruM.DruM_2D.utils.loader import load_ckpt
        from models.DruM.DruM_2D.parsers.config import get_config
        from evals.evaluate_drum import evaluate, evaluate_mol
        config = get_config(f"{args.dataset}", seed=args.seed)
        ckpt_dict = load_ckpt(config, 'cpu')
        configt = ckpt_dict['config']
        log_folder_name, log_dir, _ = set_log(configt, is_train=False)
        log_name = f"{config.ckpt}"
        log_folder_name = get_new_log_folder_name(log_folder_name)
        log_name = get_new_log_name(log_name)
        if args.dataset in ['QM9', 'ZINC250k']:
            if not os.path.exists(f"./DruM/DruM_2D/samples/mols/{log_folder_name}/{log_name}.txt"):
                print (f"./DruM/DruM_2D/samples/mols/{log_folder_name}/{log_name}.txt not found")
            else:
                gen_smiles = []
                with open (f"{log_dir}{log_name}.txt", 'r') as f:
                    for line in f:
                        gen_smiles.append(line[:-1])
                results_dict = evaluate_mol (gen_smiles, configt, config, CONSTR_CONFIG)
                print (results_dict)
        else:
            if not os.path.exists(f'./DruM/DruM_2D/samples/pkl/{log_folder_name}/{log_name}.pkl'):
                print (f'./DruM/DruM_2D/samples/pkl/{log_folder_name}/{log_name}.pkl not found')
            else:
                with open(f'./DruM/DruM_2D/samples/pkl/{log_folder_name}/{log_name}.pkl', 'rb') as f:
                    gen_graph_list = pickle.load(f)
                results_dict = evaluate (gen_graph_list, configt, config, CONSTR_CONFIG)
                print (results_dict)
    elif args.model == 'EDP-GNN':
        from models.GraphScoreMatching.utils.arg_helper import get_config
        from easydict import EasyDict as edict
        from models.GraphScoreMatching.utils.loading_utils import prepare_test_model
        from evals.evaluate_edpgnn import evaluate, evaluate_mol
        config_dict = get_config(args)
        config = edict(config_dict)
        config.save_dir = os.path.join(config.save_dir, 'sample')
        config.model_files = []
        config.init_sigma = 'inf'
        models = prepare_test_model(config)
        file, sigma_list, model_params = models[0]
        sample_dir = os.path.join(config.save_dir, 'sample_data')
        sample_dir = get_new_log_folder_name(sample_dir)
        file = get_new_log_name(file)
        if args.dataset in ['QM9', 'ZINC250k']:
            if not os.path.exists(f'./GraphScoreMatching/samples/mols/{sample_dir}/{file}.txt'):
                print (f'./GraphScoreMatching/samples/mols/{sample_dir}/{file}.txt not found')
            else:
                gen_smiles = []
                with open (f'./GraphScoreMatching/samples/mols/{sample_dir}/{file}.txt', 'r') as f:
                    for line in f:
                        gen_smiles.append(line[:-1])
                results_dict = evaluate_mol (gen_smiles, config, CONSTR_CONFIG)
                print (results_dict)
        else:
            if not os.path.exists(f'./GraphScoreMatching/samples/pkl/{sample_dir}/{file}.pkl'):
                print (f'./GraphScoreMatching/samples/pkl/{sample_dir}/{file}.pkl not found')
            else:
                with open(f'./GraphScoreMatching/samples/pkl/{sample_dir}/{file}.pkl', 'rb') as f:
                    gen_graph_list = pickle.load(f)
                results_dict = evaluate (gen_graph_list, config, CONSTR_CONFIG)
                print (results_dict)
    elif args.model == 'DiGress':
        from evals.evaluate_digress import evaluate_setting
        log_dir = f"/nethome/ksharma323/ConstrGen_Diff/DiGress/generated_samples/{args.dataset}"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_name = "generated_samples1"
        log_dir = get_new_log_folder_name(log_dir)
        log_name = get_new_log_name(log_name)
        results_dict = evaluate_setting(args.dataset, CONSTR_CONFIG, f'./DiGress/{log_dir}/{log_name}.txt')
        if results_dict == -1:
            print ('./DiGress/{log_dir}/{log_name}.txt not found')
        else:
            print (results_dict)