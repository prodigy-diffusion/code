# Diffuse, Sample, Project: Plug-And-Play Controllable Graph Generation

Supplementary code for the paper titled [Diffuse, Sample, Project: Plug-And-Play Controllable Graph Generation]().

> **TL;DR:** We present a novel sampling strategy, named PROjected DIffusion for Graphs (PRODIGY), to control the generation of graphs under well-defined constraints from any pre-trained diffusion model.

## Requirements 
We require installing the required libraries by the underlying model. In addition to that, install the additional pacakges required by our code as:

```sh
python setup.py install
```

## How to run?

```sh
python prodigy.py --model {model_name} --dataset {dataset_name} --constraint {path_to_constraint_config} --method {path_to_method_config}
```
| Args | Possibilities |
| -- | -- |
| `model_name` | [EDP-GNN, GDSS, DruM, DiGress] |
| `dataset_name` | [community_small, ego_small, enzymes, planar, sbm, QM9, ZINC250k] |

You may also add other model-specific arguments. 

**Examples**
To run the original sampling, just run 
```sh 
python prodigy.py --model {model_name} --dataset {dataset_name} 
```

You can just change the `constraint.yaml` and `method.yaml` to your desired controllable setting and run 
```sh
python prodigy.py --model {model_name} --dataset {dataset_name} --constraint constraint --method method
```


## Pre-trained Diffusion Model
We are given a graph diffusion model pre-trained on the unconditional task of matching the underlying distribution. We use the released checkpoints for each model and dataset if released by the authors or have trained our own. These can be accessed through the repositories of the corresponding models while we provide the config files for the experiments here. Checkpoints of each model used for the experiments will be released [here]() soon. 


## Constraint specification
We support a large variety of constraints and provide an implementation of the corresponding projection operators in the code base. Our paper provides a general recipe to implement the projection suitable for a constraint of your choice and can be implemented efficiently by following our code. The constraint can be flexibly specified in the `constraint.yaml` file, which allows one to adjust the following constraint parameters:

| Constraint Name | **constraint** | **params** |
| -- | -- | -- |
| Edge Count | `Num-Edges` | Maximum number of edges  $\mathcal{B}$ (ie., params: $[\mathcal{B}]$) |
| Triangle Count | `Num-Triangles` | Max number of triangles $T$ (ie., params: $[T]$) |
| Degree | `Max-Degree` | Maximum degree on any node $\delta_d$ (ie., params: $[\delta_d]$) |
| Valency | `Valency` | Valency of each atom in an ordered list (eg., [[4,3,2,1]] for [C,N,O,F]) |
| Atom Count | `Atom-Count` | Maximum Counts of each atom in an ordered list (eg., [[4,0,4,0]] for [C,N,O,F]) |
| Molecular Weight | `Mol-Weight` | Molecular weight of each atom in an ordered list followed by the maximum molecular weight of the molecule (eg., [[12, 14, 16, 19], 75]) |

One can further consider the following extensions of these constraints:
- **Box constraint:** by appending `-Box` in the **constraint** name and giving the lower bound value followed by the upper bound value in the **params**.
- **Lower Bound constraint:** by appending `-Low` in the constraint name and giving the lower bound value in the **params**.

Some representative configurations are provided in `configs/*` with the master configuration for the parameters of each constraint that we used in the experiments. 

### How to extend to an arbitrary constraint?
*[To be updated]*

We can handle arbitrary constraints by adding their appropriate projections in `projection.py`. Here, we implement 6 functions, of which you would need to update the following three functions to incorporate a new constraint:
- `constr_proj_fns(xs, adjs, constraint_config)`: 
- `project(xs, adjs, constraint_config)`:
- `satisfies(xs, adjs, constraint_config)`:

## PRODIGY parameters
Our method's parameters can be adjusted using the `method.yaml` and we explain each functionality below:

- **add_diff_step**: Additional diffusion steps over the existing pre-trained diffusion model (default: 0)
- **burnin**: Number of diffusion steps to follow the original sampling instead of PRODIGY at the start of the sampling (default: 0)
- **method**:
  - **gamma**: The gamma parameter $\gamma_t = \gamma$, will be ignored if **schedule** is set (default: 1.0).
  - **op**: Operation to do at every step (default: proj). No other operation is supported.
  <!-- - **solve_order**: cpj  -->
  <!-- - **jacobian**: false
  - **density_lambda**: 0.0
  - **last_score**: false -->
- **rounding**: Rounding scheme to round the continuous graphs to discrete space (default: none, which means random rounding). Other schemes are not supported yet and need to be implemented.
- **schedule**:
  - **gamma**: The scheduling scheme for $\gamma(t)$. Set it to `poly` for polynomial scheduling with respect to time and `exp-dist` for exponential scheduling with respect to distance. There are many other scheduling schemes that we support. 
  - **params**: Set the parameters for each scheduling scheme as a list. For example, for polynomial scheduling, set it to [a, b] to mean $\gamma(t) = $ poly $(a,b)=(1-a)t^b+a$ and for exponential scheduling, set it to $[\beta]$ to mean $\exp(- \beta \, d_{C})$. 
<!-- - implicit: false
- eq: false -->



## Plug-and-play sampler
Our method can work in a plug-and-play fashion upon any diffusion model. However, due to a lack of a standardized implementation of these methods, basic functional units differ in the code of these models. Thus, we update the official code of each paper to incorporate our projected sampling approach. These are provided in the separate codes within each repo that we call from the main file. The recipe of incorporation follows the following steps (until an in-house library is implemented). 

1. Identify the sampling function in your code. 
2. Within each iteration of the diffusion sampling, add the code for drifted_project. 

```python
"""
Sample features x from a random distribution
Sample adjacency matrix adj from a random distribution

for i in range(diff_steps):
  
  < corrector sampling >
  
  < predictor sampling > 
  
""" 
  # Add here within each diffusion iteration 
  # Assumes i to be the current iteration (0 being the first iteration)
  # and N to be the total number of diffusion iterations
  from project_bisection import drifted_project
  x, adj = drifted_project (x, adj, i, N)
```

## Evaluation
We have written specific evaluation codes for each model since they differ in what they save and how they evaluate. In particular, look at the files `evaluate_gdss.py`, `evaluate_edpgnn.py`, `evaluate_drum.py`, `evaluate_digress.py`. One can evaluate the performance of the method in `method.yaml` on `{model_name}`, `{dataset_name}` for constraint in `constraint.yaml` using:

```sh
python evaluate.py --model {model_name} --dataset {dataset_name} --constraint constraint --method method
```

## Citation

> {
>   
> }