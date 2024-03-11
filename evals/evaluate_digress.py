import os

import torch
torch.cuda.empty_cache()

from models.DiGress.src import utils
from models.DiGress.src.metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics

from models.DiGress.src.datasets.spectre_dataset import SpectreGraphDataModule
from models.DiGress.src.datasets import moses_dataset
from models.DiGress.src.metrics.molecular_metrics import TrainMolecularMetrics, SamplingMolecularMetrics
from models.DiGress.src.metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
from models.DiGress.src.analysis.spectre_utils import PlanarSamplingMetrics, SBMSamplingMetrics, Comm20SamplingMetrics
from models.DiGress.src.analysis.visualization import NonMolecularVisualization
from models.DiGress.src.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from models.DiGress.src.diffusion.extra_features_molecular import ExtraMolecularFeatures
import csv
import os
import argparse
import sys
from models.DiGress.src.utils import read_txt_graph

import yaml
from easydict import EasyDict as edict

TOTAL_MOLS = 10000
MIN_CONSTR_P = 0
seed = 42
device = 'cpu'

def get_config(dataset):
    config_dir = f'models/DiGress/configs/dataset/{dataset}.yaml'
    config = edict(yaml.load(open(config_dir, 'r'), Loader=yaml.FullLoader))
    return config

def run_setting(dataset, constr_config, filename):
        config = get_config (f'{dataset}')
        # -------- Load checkpoint --------
        
        if dataset in ['sbm', 'comm20', 'planar']:
            datamodule = SpectreGraphDataModule(edict({'dataset': config, 
                                                    'general': edict(yaml.load(open(f'../configs/general/{dataset}_default.yaml', 'r'), 
                                                                                Loader=yaml.FullLoader)),
                                                    'train': edict(yaml.load(open(f'../configs/train/train_default.yaml', 'r'), 
                                                                                Loader=yaml.FullLoader)),
                                                    'model': edict(yaml.load(open(f'../configs/model/discrete.yaml', 'r'), 
                                                                                Loader=yaml.FullLoader)),}))
            if dataset == 'sbm':
                sampling_metrics = SBMSamplingMetrics(datamodule, metrics_list=['degree', 'clustering', 'orbit', 'spectre', 'sbm', 'constr_val']) #
            elif dataset == 'comm20':
                sampling_metrics = Comm20SamplingMetrics(datamodule, metrics_list=['degree', 'clustering', 'orbit', 'constr_val'])
            elif dataset == 'planar':
                sampling_metrics = PlanarSamplingMetrics(datamodule, metrics_list=['degree', 'clustering', 'orbit', 'spectre', 'planar', 'constr_val'])
        elif dataset == 'qm9':
            cfg = edict({'dataset': config, 
                        'general': edict(yaml.load(open(f'../configs/general/qm9_default.yaml', 'r'), 
                                                    Loader=yaml.FullLoader)),
                        'train': edict(yaml.load(open(f'../configs/train/train_default.yaml', 'r'), 
                                                    Loader=yaml.FullLoader)),
                        'model': edict(yaml.load(open(f'../configs/model/discrete.yaml', 'r'), 
                                                    Loader=yaml.FullLoader)),})
            from models.DiGress.src.datasets import qm9_dataset
            datamodule = qm9_dataset.QM9DataModule(cfg)
            dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
            train_smiles = qm9_dataset.get_train_smiles(cfg=cfg, train_dataloader=datamodule.train_dataloader(),
                                                        dataset_infos=dataset_infos, evaluate_dataset=False)
            if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
                extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
                domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
            else:
                extra_features = DummyExtraFeatures()
                domain_features = DummyExtraFeatures()
                
            dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features, domain_features=domain_features)
            sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles=train_smiles)
            
        elif dataset == 'moses':
            cfg = edict({'dataset': config, 
                        'general': edict(yaml.load(open(f'../configs/general/moses_default.yaml', 'r'), 
                                                    Loader=yaml.FullLoader)),
                        'train': edict(yaml.load(open(f'../configs/train/train_default.yaml', 'r'), 
                                                    Loader=yaml.FullLoader)),
                        'model': edict(yaml.load(open(f'../configs/model/discrete.yaml', 'r'), 
                                                    Loader=yaml.FullLoader)),})
            datamodule = moses_dataset.MosesDataModule(cfg)
            dataset_infos = moses_dataset.MOSESinfos(datamodule, cfg)
            if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
                extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
                domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
            else:
                extra_features = DummyExtraFeatures()
                domain_features = DummyExtraFeatures()

            dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features, domain_features=domain_features)
            sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles=None)
            
        try:
            X, E = read_txt_graph(filename)
            out_dict = sampling_metrics(list(zip(X, E)), constraint_config=constr_config, test=True, filename=filename)
        except:
            try:
                X, E = read_txt_graph(filename)
                out_dict = sampling_metrics(list(zip(X, E)), constraint_config=constr_config, test=True)
            except:
                return -1
            
        return out_dict