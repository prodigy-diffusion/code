import argparse
import sys
import os
from project_bisection import get_new_log_name, get_new_log_folder_name, save_setting
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='constr')
    parser.add_argument('--model', type=str, default='GDSS')
    parser.add_argument('--dataset', type=str, default='community_small')
    parser.add_argument('--constraint', type=str, default='configs/none/constraint.yaml')
    parser.add_argument('--method', type=str, default='configs/none/method.yaml')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    args, _ = parser.parse_known_args(sys.argv[1:])
    args.constraint = f'{__file__}/{args.constraint}'
    args.method = f'{__file__}/{args.method}'
    print (args)
    
    if args.model == 'GDSS':
        sys.path.append(os.path.join(sys.path[0], 'models/GDSS/'))
        os.chdir('models/GDSS')
        from models.GDSS.parsers.config import get_config
        from models.GDSS.sampler import Sampler, Sampler_mol
        
        config = get_config(f'sample_{args.dataset}', args.seed)
        if config.data.data in ['QM9', 'ZINC250k']:
            sampler = Sampler_mol(config)
        else:
            sampler = Sampler(config) 
        sampler.sample()
        
        from models.GDSS.utils.logger import set_log
        from models.GDSS.utils.loader import load_ckpt
        from models.GDSS.parsers.config import get_config
        ckpt_dict = load_ckpt(config, 'cpu')
        configt = ckpt_dict['config']
        log_folder_name, log_dir, _ = set_log(configt, is_train=False)
        log_name = f"{config.ckpt}-sample"
        new_log_folder_name = get_new_log_folder_name(log_folder_name)
        new_log_dir = get_new_log_folder_name(log_dir)
        new_log_name = get_new_log_name(log_name)
        os.replace(f'./samples/pkl/{log_folder_name}/{log_name}.pkl', 
                   f'./samples/pkl/{new_log_folder_name}/{new_log_name}.pkl')
        os.replace(f'{log_dir}/{log_name}.txt', 
                   f'{new_log_dir}/{new_log_name}.txt')
        
    elif args.model == 'DruM':
        sys.path.append(os.path.join(sys.path[0], 'models/DruM/DruM_2D'))
        os.chdir('models/DruM/DruM_2D')
        
        from models.DruM.DruM_2D.parsers.config import get_config
        from models.DruM.DruM_2D.sampler import Sampler, Sampler_mol
        
        config = get_config(f'{args.dataset}', args.seed)
        if config.data.data in ['QM9', 'ZINC250k']:
            sampler = Sampler_mol(config)
        else:
            sampler = Sampler(config) 
        sampler.sample()

        from models.DruM.DruM_2D.utils.logger import set_log
        from models.DruM.DruM_2D.utils.loader import load_ckpt
        config = get_config(f"{args.dataset}", seed=args.seed)
        ckpt_dict = load_ckpt(config, 'cpu')
        configt = ckpt_dict['config']
        log_folder_name, log_dir, _ = set_log(configt, is_train=False)
        log_name = f"{config.ckpt}"
        new_log_folder_name = get_new_log_folder_name(log_folder_name)
        new_log_name = get_new_log_name(log_name)
        os.replace(f'./samples/pkl/{log_folder_name}/{log_name}.pkl', 
                   f'./samples/pkl/{new_log_folder_name}/{new_log_name}.pkl')
        os.replace(f'./samples/mols/{log_folder_name}/{log_name}.txt', 
                   f'./samples/mols/{new_log_folder_name}/{new_log_name}.txt')
    elif args.model == 'EDP-GNN':
        os.chdir('models/GraphScoreMatching/')
        from models.GraphScoreMatching.utils.arg_helper import get_config
        from models.GraphScoreMatching.sample import sample_main
        config_dict = get_config(args)
        sample_main(config_dict, args)
        
        from easydict import EasyDict as edict
        from models.GraphScoreMatching.utils.loading_utils import prepare_test_model
        config_dict = get_config(args)
        config = edict(config_dict)
        config.save_dir = os.path.join(config.save_dir, 'sample')
        config.model_files = []
        config.init_sigma = 'inf'
        models = prepare_test_model(config)
        file, sigma_list, model_params = models[0]
        sample_dir = os.path.join(config.save_dir, 'sample_data')
        new_sample_dir = get_new_log_folder_name(sample_dir)
        new_file = get_new_log_name(file)
        os.replace(f'./samples/pkl/{sample_dir}/{file}.pkl', 
                   f'./samples/pkl/{new_sample_dir}/{new_file}.pkl')
        os.replace(f'./samples/mols/{sample_dir}/{file}.txt', 
                   f'./samples/mols/{new_sample_dir}/{new_file}.txt')
    elif args.model == 'DiGress':
        os.chdir('models/DiGress/src')
        from models.DiGress.src import main
        main({'dataset': f'{args.dataset}', 'general': f'{args.dataset}_default'})

        log_dir = f"/nethome/ksharma323/ConstrGen_Diff/DiGress/generated_samples/{args.dataset}"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_name = "generated_samples1"
        new_log_dir = get_new_log_folder_name(log_dir)
        new_log_name = get_new_log_name(log_name)
        os.replace(f'./{log_dir}/{log_name}.txt', 
                   f'./{new_log_dir}/{new_log_name}.txt')
        # change the default logger name to some other name based on constraint and method
