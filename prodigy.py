import argparse
import sys
import os
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
        os.chdir('GDSS/')
        os.system(f"python main.py --type sample --config sample_{args.dataset} --device {args.device} --seed {args.seed}")
        from models.GDSS.utils.logger import set_log
        from models.GDSS.utils.loader import load_ckpt
        from models.GDSS.parsers.config import get_config
        config = get_config(f"sample_{args.dataset}", seed=args.seed)
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
        os.chdir('DruM/DruM_2D')
        os.system(f"python main.py --type sample --config {args.dataset} --seed {args.seed}")
        from models.DruM.DruM_2D.utils.logger import set_log
        from models.DruM.DruM_2D.utils.loader import load_ckpt
        from models.DruM.DruM_2D.parsers.config import get_config
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
        os.chdir('GraphScoreMatching/')
        os.system(f"python sample.py --dataset {args.dataset} --given_param")
        from models.GraphScoreMatching.utils.arg_helper import get_config
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
        os.chdir('DiGress/')
        os.system(f"python main.py dataset={args.dataset} general={args.dataset}_default")
        log_dir = f"/nethome/ksharma323/ConstrGen_Diff/DiGress/generated_samples/{args.dataset}"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_name = "generated_samples1"
        new_log_dir = get_new_log_folder_name(log_dir)
        new_log_name = get_new_log_name(log_name)
        os.replace(f'./{log_dir}/{log_name}.txt', 
                   f'./{new_log_dir}/{new_log_name}.txt')
        # change the default logger name to some other name based on constraint and method
