import csv
import os
from easydict import EasyDict as edict
import yaml
import sys
import argparse

def get_constraint_config(config):
    if '/' not in config:
        config_name = f'{__file__}/config/constraints/{config}.yaml'
    else:
        config_name = f'{config}'
    config = edict(yaml.load(open(config_name, 'r'), Loader=yaml.FullLoader))
    return config

def get_method_config(config):
    if '/' not in config:
        config_name = f'{__file__}/config/constraints/{config}.yaml'
    elif '.yaml' not in config:
        config_name = f'{config}.yaml'
    else:
        config_name = f'{config}'
    config = edict(yaml.load(open(config_name, 'r'), Loader=yaml.FullLoader))
    return config


def save_setting(results_dict, model, dataset, constraint):
    results_dict['dataset'] = dataset
    results_dict['constraint'] = constraint
    results_dict['model'] = model
    with open (f"eval.csv", "a") as wf, open (f"eval.csv", "a") as mwf:
        writer = csv.DictWriter(wf, fieldnames=['model', 'dataset', 'constraint', 'param', 'method', 'method_param', 'degree', 'cluster', 'orbit', 'spectral', 'constr_val'])
        mwriter = csv.DictWriter(mwf, fieldnames=['model', 'dataset', 'constraint', 'param', 'method', 'method_param', 'valid', 'unique', 'fcd', 'novelty', 'nspdk', 'num_mols', 'constr_val'])
        if dataset in ['qm9', 'zinc250k', 'moses']:
            mwf.write(','.join([f'{x}:{y}' for x, y in results_dict.items()]) + '\n')
            # mwriter.writerow(results_dict)
        else:
            writer.writerow(results_dict)
                      
def get_new_log_folder_name(log_folder_name):
    log_folder_name = f"{log_folder_name}/{CONSTR_CONFIG.constraint}"
    if not os.path.exists(log_folder_name):
        os.makedirs(log_folder_name)
    return log_folder_name

def get_new_log_name(log_name):
    param_vals = map (str, CONSTR_CONFIG.params)
    log_name += f"-{','.join(param_vals)}"
    log_name += f"-{METHOD_CONFIG.method.op}-{METHOD_CONFIG.method.gamma}"
    log_name += f"-{METHOD_CONFIG.schedule.gamma}{','.join(map(str, METHOD_CONFIG.schedule.params))}"
    log_name += f"-{METHOD_CONFIG.burnin}"
    log_name += f"-{METHOD_CONFIG.add_diff_step}"
    log_name += f"-{METHOD_CONFIG.rounding}"
    log_name = log_name.replace(".", "p")


parser = argparse.ArgumentParser(description='constr')
parser.add_argument('--constr_config', type=str, default='configs/none/method.yaml')
parser.add_argument('--method_config', type=str, default='configs/none/constraint.yaml')
args = parser.parse_known_args(sys.argv[1:])
CONSTR_CONFIG = get_constraint_config(args.constr_config)
METHOD_CONFIG = get_method_config(args.method_config)
