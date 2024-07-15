import torch
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
parser.add_argument('--constraint', type=str, default='configs/none/method.yaml')
parser.add_argument('--method', type=str, default='configs/none/constraint.yaml')
args, _ = parser.parse_known_args(sys.argv[1:])

args.constraint = f'{os.path.dirname(__file__)}/{args.constraint}' if not args.constraint.startswith('/') else args.constraint
args.method = f'{os.path.dirname(__file__)}/{args.method}' if not args.method.startswith('/') else args.method

CONSTR_CONFIG = get_constraint_config(args.constraint)
METHOD_CONFIG = get_method_config(args.method)

P01 = lambda y: torch.clamp (y, min=0, max=1)
P03 = lambda y: torch.clamp (y, min=0, max=3)

def project_sublevel (inputs, proj_fn, constr_fn, mus_lower, mus_upper, constr_params=None, eq=False, diff_tol=1e-5, max_iters=100):
    # constr_fn looks like an upperbound constr(proj, mu) - constr_params <= 0
    # constr_fn is broadcasted over the inputs
    if eq:
        sat_mask = constr_fn (proj_fn (inputs, params=constr_params), params=constr_params).abs() <= diff_tol
    else:
        sat_mask = constr_fn (proj_fn (inputs, params=constr_params), params=constr_params) <= 0
    proj_inputs = inputs.clone()
    if torch.any(sat_mask):
        proj_inputs[sat_mask] = proj_fn (inputs[sat_mask], params=constr_params)
    # bisection, find mu
    all_mus = torch.zeros_like(mus_lower)
    if (torch.all(sat_mask)):
        return proj_inputs, all_mus
    unsat_inputs = proj_inputs[~sat_mask]
    mus_lower = mus_lower[~sat_mask]
    mus_upper = mus_upper[~sat_mask]
    params = constr_params[~sat_mask] if (type(constr_params) is torch.Tensor) else constr_params
    for i in range(max_iters):
        mus = (mus_lower + mus_upper) / 2.0
        # print ((constr_fn (proj_fn (unsat_inputs, mus=mus_lower)) * constr_fn (proj_fn (unsat_inputs, mus=mus_upper)) < 0).sum())
        # print (i, mus_lower, mus_upper, constr_fn (proj_fn (unsat_inputs, mus=mus, params=params), params=params).abs())
        # still_bisect = ((torch.abs(mus_upper - mus_lower) > diff_tol) & 
        #                         (constr_fn (proj_fn (unsat_inputs, mus=mus, params=params), params=params).abs() > diff_tol))
        still_bisect = (constr_fn (proj_fn (unsat_inputs, mus=mus, params=params), params=params).abs() > diff_tol)
        if torch.all (~still_bisect):
            break
        upper_update = torch.zeros_like (still_bisect)
        # upper_update if it's in a different direction.
        params_sb = params[still_bisect] if (type(constr_params) is torch.Tensor) else params
        upper_update[still_bisect] = ((constr_fn (proj_fn (unsat_inputs[still_bisect], mus=mus_lower[still_bisect], params=params_sb), params=params_sb) *
                                       constr_fn (proj_fn (unsat_inputs[still_bisect], mus=mus[still_bisect], params=params_sb), params=params_sb)) < 0)
        mus_upper[still_bisect & upper_update] = mus[still_bisect & upper_update]
        mus_lower[still_bisect & ~upper_update] = mus[still_bisect & ~upper_update]
    proj_inputs[~sat_mask] = proj_fn (unsat_inputs, mus=mus, params=params)
    all_mus[~sat_mask] = mus
    return proj_inputs, all_mus

def find_muUpper (inputs, proj_fn, constr_fn, mus_lower, params, max_iters=100):
    mus_upper = mus_lower.clone()
    step_size = 0.5
    still_find = torch.ones_like (mus_lower, dtype=bool)
    params_sf = params[still_find] if type(params) is torch.Tensor else params
    # print (proj_fn (inputs[still_find], mus=mus_lower[still_find], params=params_sf))
    still_find[(constr_fn (proj_fn (inputs[still_find], mus=mus_lower[still_find], params=params_sf), params=params_sf) <= 0)] = False
    if ~torch.any(still_find):
        return mus_upper
    mus_upper_iters = mus_upper.clone()
    for i in range (max_iters):
        params_sf = params[still_find] if type(params) is torch.Tensor else params
        # print (i, constr_fn (proj_fn (inputs[still_find], mus=mus_lower[still_find], params=params_sf), params=params_sf),
        #         constr_fn (proj_fn (inputs[still_find], mus=mus_upper_iters[still_find], params=params_sf), params=params_sf))
        still_find.scatter_ (dim=0, index=torch.where(still_find)[0], 
                             src=((constr_fn (proj_fn (inputs[still_find], mus=mus_lower[still_find], params=params_sf), params=params_sf) *
                                   constr_fn (proj_fn (inputs[still_find], mus=mus_upper_iters[still_find], params=params_sf), params=params_sf)) > 0))
        if ~torch.any(still_find):
            break
        # print (i, torch.where(still_find)[0])
        mus_upper_iters[still_find] += step_size
    mus_upper[~still_find] = mus_upper_iters[~still_find]
    return mus_upper

def constr_proj_fns (xs, adjs, constraint_config):
    # Proj(input)
    P01 = lambda y: torch.clamp (y, min=0, max=1)
    P03 = lambda y: torch.clamp (y, min=0, max=3)
    Pleaky01 = lambda y: torch.clamp (y, min=1e-5, max=1)
    constr_fn_above, proj_fn_above = None, None
    if 'Num-Edges-EQ' in constraint_config.constraint:
        # converting to vector
        constr_fn = lambda val: lambda avs, params=None, **kwargs: (avs.sum(dim=1) - (val[params] if params is not None else val))
        proj_fn = lambda avs, mus=None, **kwargs: P01 (avs - mus[:, None]) if mus is not None else P01 (avs)
    elif 'Num-Triangles-EQ' in constraint_config.constraint:
        constr_fn = lambda val: lambda As, params=None, **kwargs: (1/6 * torch.diagonal(torch.matrix_power(As, 3), dim1=1, dim2=2).sum(dim=1) - (val[params] if params is not None else val))
        proj_fn = lambda As, mus=None, **kwargs: P01 (As - mus[:, None, None] * torch.matrix_power(Pleaky01(As), 2)) if mus is not None else P01 (As)
    elif 'Num-Edges' in constraint_config.constraint:
        # converting to vector
        constr_fn = lambda val: lambda avs, **kwargs: (avs.sum(dim=1) - val)
        proj_fn = lambda avs, mus=None, **kwargs: P01 (avs - mus[:, None]) if mus is not None else P01 (avs)
    elif 'Num-Triangles' in constraint_config.constraint:
        constr_fn = lambda val: lambda As, **kwargs: (1/6 * torch.diagonal(torch.matrix_power(As, 3), dim1=1, dim2=2).sum(dim=1) - val)
        proj_fn = lambda As, mus=None, **kwargs: P01 (As - mus[:, None, None] * torch.matrix_power(Pleaky01(As), 2)) if mus is not None else P01 (As)
    elif 'Max-Degree' in constraint_config.constraint:
        constr_fn_above = lambda val: lambda As, **kwargs: (As.sum(dim=2) - val)
        proj_fn_above = lambda As, mus=None, **kwargs: P01 (As - (mus[:, :, None] + mus[:, None, :]) + 2*torch.stack([torch.diag(mui) for mui in mus])) if mus is not None else P01(As)
        # find mu~ 
        constr_fn = lambda val: lambda avs, **kwargs: (avs.sum(dim=1) - val)
        proj_fn = lambda avs, mus=None, **kwargs: P01 (avs - mus[:, None]) if mus is not None else P01 (avs)
    elif 'Valency' in constraint_config.constraint:
        valencies = torch.tensor(constraint_config.params[0], dtype=xs.dtype, device=xs.device)
        proj_xs = P01(xs)
        wtd_vals = proj_xs @ valencies
        constr_fn_above  = lambda val: lambda As, **kwargs: (As.sum(dim=2) - wtd_vals.reshape(As.shape[0], As.shape[1]))
        proj_fn_above = lambda As, mus=None, **kwargs: P03 (As - (mus[:, :, None] + mus[:, None, :]) + 2*torch.stack([torch.diag(mui) for mui in mus])) if mus is not None else P03(As)
        constr_fn = lambda val: lambda avs, params, **kwargs: avs.sum(dim=1) - params
        proj_fn = lambda avs, mus=None, **kwargs: P03 (avs - mus[:, None]) if mus is not None else P03 (avs)
    elif 'Atom-Count' in constraint_config.constraint:
        constr_fn = lambda val: lambda xTs, params=None, **kwargs: (xTs.sum(dim=1) - params)
        proj_fn = lambda xTs, mus=None, **kwargs: P01 (xTs - mus[:, None]) if mus is not None else P01 (xTs)
    elif 'Mol-Weight' in constraint_config.constraint:
        atomWeights = torch.tensor (constraint_config.params[0], dtype=xs.dtype, device=xs.device)
        constr_fn = lambda val: lambda Xs, **kwargs: (Xs @ atomWeights).sum(dim=1) - val
        proj_fn = lambda Xs, mus=None, **kwargs: P01 (Xs - (mus[:, None] @ atomWeights[None, :])[:, None, :]) if mus is not None else P01 (Xs)
    elif 'Property-MLP' in constraint_config.constraint:
        xtheta_params = torch.load (constraint_config.params[0], map_location=xs.device)
        atheta_params = torch.load (constraint_config.params[1], map_location=xs.device)
        def constr_fn (xas, **kwargs):
            xvecs = xas[:, :xtheta_params.shape[0]]
            avecs = xas[:, xtheta_params.shape[0]:]
            return lambda val: xtheta_params @ xvecs.T + atheta_params @ avecs.T - val
        def proj_fn (xas, mus=None, **kwargs):
            xvecs = xas[:, :xtheta_params.shape[0]]
            avecs = xas[:, xtheta_params.shape[0]:]
            if mus is not None:
                return torch.cat ((P01 (xvecs - (mus[:, None] @ xtheta_params[None, :])), P03 (avecs - (mus[:, None] @ atheta_params[None, :]))), dim=1)
            else:
                return torch.cat ((P01 (xvecs), P03 (avecs)), dim=1)
    elif 'Property-SGC-In' in constraint_config.constraint:
        theta_params = torch.load (f"config/constraints/regmodels/{constraint_config.params[0]}", map_location=xs.device)
        theta = theta_params['conv.lin.weight'].squeeze().type(xs.dtype).to(xs.device)
        bias = theta_params['conv.lin.bias'].item()
        constr_fn = lambda val: lambda Xs, params=None, **kwargs: ((params @ Xs @ theta[None, :, None])[:,:,0].mean(dim=1) + bias - val)
        proj_fn = lambda Xs, mus=None, params=None, **kwargs: (P01 (Xs - mus[:, None, None] * (params.sum(dim=1)[:, :, None] @ theta[None, None, :])) if mus is not None else P01 (Xs))
    return constr_fn, proj_fn, constr_fn_above, proj_fn_above


def project (xs, adjs, constraint_config):
    # eq = constraint_config.eq if "eq" in constraint_config else False
    Pleaky01 = lambda y: torch.clamp (y, min=1e-5, max=1)
    constr_fn, proj_fn, constr_fn_above, proj_fn_above = constr_proj_fns (xs, adjs, constraint_config)
    constr_params = None
    if 'Num-Edges' in constraint_config.constraint:
        if constraint_config.params[0] == 'zeros':
            adjs0 = torch.zeros_like (adjs)
        else:
            adjs0 = torch.load (constraint_config.params[0])
        values = constraint_config.params[1:]
        row_inds, col_inds = torch.triu_indices(adjs.shape[1], adjs.shape[2], offset=1)
        inps = (adjs - adjs0)[:, row_inds, col_inds]
        mus1 = inps.reshape (inps.shape[0], -1).min(dim=1)[0] - 1
        mus1 = torch.clamp (mus1, min=0)
        mus2 = inps.reshape (inps.shape[0], -1).max(dim=1)[0]
        def reverse_transf (proj_inps, proj_mus=None):
            proj_adjs = torch.zeros_like (adjs)
            proj_adjs[:, row_inds, col_inds] = adjs0[:, row_inds, col_inds] + proj_inps
            proj_adjs = proj_adjs + proj_adjs.transpose (1, 2)
            return xs, proj_adjs
    elif 'Num-Triangles' in constraint_config.constraint:
        values = constraint_config.params
        inps = adjs
        adjs2 = torch.matrix_power(Pleaky01(adjs), 2)
        adjs_vecs, adjs2_vecs = adjs.reshape(adjs.shape[0], -1), adjs2.reshape(adjs2.shape[0], -1)
        mus1 = torch.clamp(2*adjs_vecs.min(dim=1)[0]/adjs2_vecs.max(dim=1)[0] - 1/adjs2_vecs.min(dim=1)[0], 0)
        mus2 = 2*adjs_vecs.max(dim=1)[0]/adjs2_vecs.min(dim=1)[0]
        reverse_transf = lambda proj_inps, proj_mus=None: (xs, proj_inps)
    elif 'Max-Degree' in constraint_config.constraint:
        values = constraint_config.params
        inps = adjs.reshape(-1, adjs.shape[1])
        mus1 = inps.min(dim=1)[0] - 1
        mus1 = torch.clamp (mus1, min=0)
        mus2 = inps.max(dim=1)[0]
        reverse_transf = lambda proj_inps, proj_mus=None: (xs, proj_fn_above (adjs, proj_mus.reshape(adjs.shape[0], adjs.shape[1])))
    elif 'Valency' in constraint_config.constraint:
        proj_xs = P01(xs)
        values = [None]
        valencies = torch.tensor (constraint_config.params[0], dtype=xs.dtype, device=xs.device)
        wtd_vals = proj_xs @ valencies
        constr_params = wtd_vals.reshape(-1)
        inps = adjs.reshape(-1, adjs.shape[1])
        mus1 = inps.min(dim=1)[0] - 1
        mus1 = torch.clamp (mus1, min=0)
        mus2 = inps.max(dim=1)[0]
        reverse_transf = lambda proj_inps, proj_mus=None: (proj_xs, proj_fn_above (adjs, proj_mus.reshape(adjs.shape[0], adjs.shape[1])))
    elif 'Atom-Count' in constraint_config.constraint:
        # values = constraint_config.params
        values = torch.tensor (constraint_config.params[0], dtype=xs.dtype, device=xs.device).repeat(xs.shape[0])
        constr_params = values
        values = [None]
        inps = xs.transpose(1, 2).reshape(-1, xs.shape[1])
        mus1 = torch.clamp(inps.min(dim=1)[0] - 1, min=0)
        mus2 = inps.max(dim=1)[0]
        reverse_transf = lambda proj_inps, proj_mus=None: (proj_inps.reshape(xs.shape[0], xs.shape[2], xs.shape[1]).transpose(1, 2), adjs)
    elif 'Mol-Weight' in constraint_config.constraint:
        atomWeights = torch.tensor (constraint_config.params[0], dtype=xs.dtype, device=xs.device)
        values = constraint_config.params[1:]
        inps = xs
        mus1 = torch.clamp(xs.reshape(xs.shape[0], -1).min(dim=1)[0]/atomWeights.max() - 1/atomWeights.min(), min=0)
        mus2 = xs.reshape(xs.shape[0], -1).max(dim=1)[0]
        reverse_transf = lambda proj_inps, proj_mus=None: (proj_inps, adjs)
    elif 'Property-MLP' in constraint_config.constraint:
        xtheta_params = torch.load (constraint_config.params[0], map_location=xs.device)
        values = constraint_config.params[2:]
        x_vecs = xs.reshape (xs.shape[0], -1)
        adj_vecs = adjs.reshape (adjs.shape[0], -1)
        inps = torch.cat ((x_vecs, adj_vecs), dim=1)
        mus1 = torch.zeros (inps.shape[0], dtype=inps.dtype, device=inps.device)
        mus2 = None
        reverse_transf = lambda proj_inps, proj_mus=None: (proj_inps[:, :xtheta_params.shape[0]].reshape(xs.shape), proj_inps[:, xtheta_params.shape[0]:].reshape(adjs.shape))
    elif 'Property-SGC' in constraint_config.constraint:
        values = constraint_config.params[1:]
        nlayers = int(constraint_config.params[0].split("_")[0][3:])
        N, n = adjs.shape[0], adjs.shape[1]
        proj_adjs = P03 (adjs)
        adjs_norm = proj_adjs.clone()
        adjs_norm[:, torch.arange(n), torch.arange(n)] = 1
        degs_norm = adjs_norm.sum(dim=2)
        degs_norm = (degs_norm[:, :, None] @ degs_norm[:, None, :])**0.5
        adjs_norm = adjs_norm / degs_norm
        adjs_norm_k = torch.matrix_power(adjs_norm, nlayers)
        constr_params = adjs_norm_k
        inps = xs
        mus1 = torch.zeros (xs.shape[0], dtype=xs.dtype, device=xs.device)
        mus2 = None
        reverse_transf = lambda proj_inps, proj_mus=None: (proj_inps, proj_adjs)

    if 'EQ' in constraint_config.constraint:
        all_values = torch.tensor(values[0]).to(adjs.device)
        unique_vals, unique_counts = torch.unique (all_values, return_counts=True)
        values = unique_vals[torch.multinomial (unique_counts.float(), adjs.shape[0], replacement=True)]
        constr_params = torch.arange(len(values)).to(adjs.device)
        val_low, val_upp = values, values
        projfn_low = lambda x, mus=None, **kwargs: proj_fn(x, mus=-mus, **kwargs) if mus is not None else proj_fn(x, **kwargs)
        projfn_upp = proj_fn
        constrfn_low = lambda x, params=None, **kwargs: - constr_fn (val_low)(x, params=params, **kwargs)
        constrfn_upp = constr_fn (val_upp)
        mus1_low = mus1
        mus2_low = find_muUpper (inps, projfn_low, constrfn_low, mus1, params=constr_params) if mus2 is None else mus2
        mus1_upp = mus1
        mus2_upp = find_muUpper (inps, projfn_upp, constrfn_upp, mus1, params=constr_params) if mus2 is None else mus2
        projLow_inps, projLow_mus = project_sublevel (inps, projfn_low, constrfn_low, mus1_low, mus2_low, constr_params=constr_params)
        # unsat_mask_low = constrfn_low (projfn_low (inps)) > 0
        proj_inps, proj_mus = project_sublevel (projLow_inps, projfn_upp, constrfn_upp, mus1_upp, mus2_upp, constr_params=constr_params)
        # proj_inps, proj_mus = project_sublevel (inps, projfn_upp, constrfn_upp, mus1_upp, mus2_upp, constr_params=constr_params)
        # proj_inps[unsat_mask_low] = projLow_inps[unsat_mask_low]
        # proj_mus[unsat_mask_low] = projLow_mus[unsat_mask_low]
    elif 'Box' in constraint_config.constraint:
        # Box constraint
        assert (len(values) >= 2)
        val_low, val_upp = values[0], values[1]
        assert (val_low <= val_upp)
        projfn_low = lambda x, mus=None, **kwargs: proj_fn(x, mus=-mus, **kwargs) if mus is not None else proj_fn(x, **kwargs)
        projfn_upp = proj_fn
        constrfn_low = lambda x, params=None, **kwargs: - constr_fn (val_low)(x, params=params, **kwargs)
        constrfn_upp = constr_fn (val_upp)
        mus1_low = mus1
        mus2_low = find_muUpper (inps, projfn_low, constrfn_low, mus1, params=constr_params) if mus2 is None else mus2
        mus1_upp = mus1
        mus2_upp = find_muUpper (inps, projfn_upp, constrfn_upp, mus1, params=constr_params) if mus2 is None else mus2
        projLow_inps, projLow_mus = project_sublevel (inps, projfn_low, constrfn_low, mus1_low, mus2_low, constr_params=constr_params)
        unsat_mask_low = constrfn_low (projfn_low (inps)) > 0
        proj_inps, proj_mus = project_sublevel (inps, projfn_upp, constrfn_upp, mus1_upp, mus2_upp, constr_params=constr_params)
        proj_inps[unsat_mask_low] = projLow_inps[unsat_mask_low]
        proj_mus[unsat_mask_low] = projLow_mus[unsat_mask_low]
    elif 'Low' in constraint_config.constraint:
        # Lower-bound constraint
        projfn_low = lambda inps, mus=None, **kwargs: proj_fn(inps, mus=-mus, **kwargs) if mus is not None else proj_fn(inps, **kwargs)
        constrfn_low = lambda inps, params=None, **kwargs: - constr_fn(values[0])(inps, params=params, **kwargs)
        mus1 = mus1
        mus2 = find_muUpper (inps, projfn_low, constrfn_low, mus1, params=constr_params) if mus2 is None else mus2
        proj_inps, proj_mus = project_sublevel (inps, projfn_low, constrfn_low, mus1, mus2, constr_params=constr_params)
    else:
        # Upper-bound constraint
        mus2 = find_muUpper (inps, proj_fn, constr_fn(values[0]), mus1, params=constr_params) if mus2 is None else mus2
        proj_inps, proj_mus = project_sublevel (inps, proj_fn, constr_fn(values[0]), mus1, mus2, constr_params=constr_params)

    return reverse_transf (proj_inps, proj_mus)


def drifted_project (xs, adjs, i, diff_steps, constraint_config=None, method_config=None):
    constraint_config = CONSTR_CONFIG if constraint_config is None else constraint_config
    method_config = METHOD_CONFIG if method_config is None else method_config
    if method_config.schedule.gamma == "cyclical":
        if i % method_config.schedule.params[0] != 0:
            return xs, adjs
    elif method_config.schedule.gamma == "step":
        method_config.method.gamma = 1 if i >= method_config.schedule.params[0] else 0
    elif method_config.schedule.gamma == "steprise":
        # not useful as with gamma_mulp = 2, 0.5 ---> 1 in one step.
        # not smooth gradation
        gamma_init, gamma_mulp, gamma_steps = method_config.schedule.params
        method_config.method.gamma = min(1., gamma_init * gamma_mulp * (i//gamma_steps + 1))
    elif method_config.schedule.gamma == "poly":
        # smooth (1-gamma_init) * t^k + gamma_init
        # assumed to reach 1 at the end (i.e., t = 1)
        gamma_init, time_pow = method_config.schedule.params
        method_config.method.gamma = (1 - gamma_init) * (i/diff_steps)**time_pow + gamma_init
    elif method_config.schedule.gamma == "poly-end":
        # smooth (1-gamma_init) * t^k + gamma_init
        # assumed to reach 1 at the end (i.e., t = 1)
        gamma_end, time_pow = method_config.schedule.params
        method_config.method.gamma = gamma_end * (i/diff_steps)**time_pow
    elif method_config.schedule.gamma == "polystep":
        # smooth (1-gamma_init) * t^k + gamma_init
        # assumed to reach 1 at the end (i.e., t = 1)
        gamma_init, time_pow, gamma_step = method_config.schedule.params
        method_config.method.gamma = (1 - gamma_init) * (i//gamma_step*(gamma_step/diff_steps))**time_pow + gamma_init
    elif method_config.schedule.gamma == "polymid":
        # smooth (1-gamma_init) * t^k + gamma_init
        # reaches 1 at some step in the middle 
        gamma_init, time_pow, one_step = method_config.schedule.params
        if i >= one_step:
            method_config.method.gamma = 1.
        else:
            method_config.method.gamma = (1 - gamma_init) * (i/diff_steps)**time_pow/((one_step/diff_steps)**time_pow) + gamma_init
    elif method_config.schedule.gamma == "fixed":
        method_config.method.gamma = method_config.schedule.params[0]
    elif method_config.schedule.gamma == 'exp-dist':
        proj_xs, proj_adjs = project (xs, adjs, constraint_config)
        adj_dist = (proj_adjs - adjs).reshape(adjs.shape[0], -1).norm(dim=1, p=2) / (adjs.shape[1] * adjs.shape[2])
        x_dist = (proj_xs - xs).reshape(xs.shape[0], -1).norm(dim=1, p=2) / (xs.shape[1] * xs.shape[2])
        thresh, beta = method_config.schedule.params
        drift_x = torch.where(x_dist < thresh, torch.ones_like(x_dist), torch.exp(-(x_dist - thresh) * beta))
        drift_adj = torch.where(adj_dist < thresh, torch.ones_like(x_dist), torch.exp(-(adj_dist - thresh) * beta))
        return xs + drift_x[:, None, None] * (proj_xs - xs), adjs + drift_adj[:, None, None] * (proj_adjs - adjs)
    
    drift = method_config.method.gamma if "method" in constraint_config else 1
    if "method" in constraint_config and drift == 0:
        return xs, adjs
    proj_xs, proj_adjs = project (xs, adjs, constraint_config)
    return xs + drift * (proj_xs - xs), adjs + drift * (proj_adjs - adjs)


def satisfies (xs, adjs, constraint_config, zero_tol=1e-5):
    # eq = constraint_config.eq if "eq" in constraint_config else False
    if 'None' in constraint_config.constraint:
        return torch.ones (adjs.shape[0], dtype=bool, device=adjs.device)     
    constr_fn, _, constr_fn_above, _ = constr_proj_fns (xs, adjs, constraint_config)
    constr_params = None
    if 'Num-Edges' in constraint_config.constraint:
        if constraint_config.params[0] == 'zeros':
            adjs0 = torch.zeros_like (adjs)
        else:
            adjs0 = torch.load (constraint_config.params[0])
        values = constraint_config.params[1:]
        row_inds, col_inds = torch.triu_indices(adjs.shape[1], adjs.shape[2], offset=1)
        inps = (adjs - adjs0)[:, row_inds, col_inds]
    elif 'Num-Triangles' in constraint_config.constraint:
        values = constraint_config.params
        inps = adjs
    elif 'Max-Degree' in constraint_config.constraint:
        values = constraint_config.params
        inps = adjs
        constr_fn = constr_fn_above
    elif 'Valency' in constraint_config.constraint:
        # check
        proj_xs = P01(xs)
        values = [None]
        valencies = torch.tensor (constraint_config.params[0], dtype=xs.dtype, device=xs.device)
        wtd_vals = proj_xs @ valencies
        constr_params = wtd_vals.reshape(-1)
        inps = adjs
        constr_fn = constr_fn_above
    elif 'Atom-Count' in constraint_config.constraint:
        values = [None]
        # values = torch.tensor (constraint_config.params[0], dtype=xs.dtype, device=xs.device).repeat(xs.shape[0])
        constr_params = torch.tensor (constraint_config.params[0], dtype=xs.dtype, device=xs.device).repeat(xs.shape[0], 1)
        inps = xs
    elif 'Mol-Weight' in constraint_config.constraint:
        atomWeights = torch.tensor (constraint_config.params[0], dtype=xs.dtype, device=xs.device)
        values = constraint_config.params[1:]
        inps = xs
    elif 'Property-MLP' in constraint_config.constraint:
        values = constraint_config.params[2:]
        x_vecs = xs.reshape (xs.shape[0], -1)
        adj_vecs = adjs.reshape (adjs.shape[0], -1)
        inps = torch.cat ((x_vecs, adj_vecs), dim=1)
    elif 'Property-SGC' in constraint_config.constraint:
        values = constraint_config.params[1:]
        nlayers = int(constraint_config.params[0].split("_")[0][3:])
        N, n = adjs.shape[0], adjs.shape[1]
        proj_adjs = P03 (adjs)
        adjs_norm = proj_adjs.clone()
        adjs_norm[:, torch.arange(n), torch.arange(n)] = 1
        degs_norm = adjs_norm.sum(dim=2)
        degs_norm = (degs_norm[:, :, None] @ degs_norm[:, None, :])**0.5
        adjs_norm = adjs_norm / degs_norm
        adjs_norm_k = torch.matrix_power(adjs_norm, nlayers)
        constr_params = adjs_norm_k
        inps = xs

    if 'EQ' in constraint_config.constraint:
        all_values = torch.tensor(values[0]).to(adjs.device)
        unique_vals, unique_counts = torch.unique (all_values, return_counts=True)
        values = unique_vals[torch.multinomial (unique_counts.float(), adjs.shape[0], replacement=True)]
        val_low, val_upp = values, values
        constr_sats = constr_fn(val_low)(inps, params=constr_params)
        constr_low = (constr_sats >= 0) | (constr_sats.abs() <= zero_tol)
        constr_sats = constr_fn(val_upp)(inps, params=constr_params)
        constr_upp = (constr_sats <= 0) | (constr_sats.abs() <= zero_tol)
        agg_constr_sats = (constr_low & constr_upp)
    elif 'Box' in constraint_config.constraint:
        # Box constraint
        assert (len(values) >= 2)
        val_low, val_upp = values[0], values[1]
        assert (val_low <= val_upp)
        constr_sats = constr_fn(val_low)(inps, params=constr_params)
        constr_low = (constr_sats >= 0) | (constr_sats.abs() <= zero_tol)
        constr_sats = constr_fn(val_upp)(inps, params=constr_params)
        constr_upp = (constr_sats <= 0) | (constr_sats.abs() <= zero_tol)
        agg_constr_sats = (constr_low & constr_upp)
    elif 'Low' in constraint_config.constraint:
        # Lower-bound constraint
        constr_sats = constr_fn(values[0])(inps, params=constr_params)
        agg_constr_sats = (constr_sats >= 0) | (constr_sats.abs() <= zero_tol)
    else:
        # Upper-bound constraint
        constr_sats = constr_fn(values[0])(inps, params=constr_params)
        agg_constr_sats = (constr_sats <= 0) | (constr_sats.abs() <= zero_tol)
    return agg_constr_sats if agg_constr_sats.ndim == 1 else torch.all (agg_constr_sats, dim=1)
