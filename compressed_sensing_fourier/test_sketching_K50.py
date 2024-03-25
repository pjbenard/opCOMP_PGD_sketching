# Import sys for accessing solversuperres_v3 package
import sys
sys.path.insert(0, '..')

import argparse

# Import basic packages
import numpy as np
import numpy.linalg as npl

from tqdm import tqdm
from functools import partial
from typing import Callable, Tuple
from itertools import product
from time import perf_counter_ns

# Import custom solversuperres package
from solversuperres_v3.data_utils        import semi_gridded_init
from solversuperres_v3.g_utils           import gradient_g, g
from solversuperres_v3.descent_utilities import clip_domain, projection_X
from solversuperres_v3.opCOMP_init       import opCOMP

import solversuperres_v3.FISTA_restart_descent_v2 as FISTA
import solversuperres_v3.init_utils               as init

from solversuperres_v3.linop_Gaussian_MATIRF        import           Gaussian2D_MATIRF
from solversuperres_v3.linop_Gaussian_MATIRF_Sketch import Sketching_Gaussian2D_MATIRF

# Parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument('part', type=int)
args = parser.parse_args()

part = args.part
###################

print(part)
NREP = 1
r_NREP_p = range((part - 1) * NREP, part * NREP)
measures_per_spike = list(range(3, 52, 3))

SAVE_DATA = True

# Global parameters
d = 3
N = 50
density =np.array([5, 5, 2])

disable_tqdm = True
###################
# Parameters OP-COMP
step_opCOMP = 1e-4
nb_tests = 1
descent_nit_OP_COMP = 1_000
tol_criterion = 0.08
multicore = False

###################
# Parameters PGD
step_PGD = 1e-4
descent_nit_PGD = 75_000
abs_tol_cst=1e-8 
rel_tol_cst=1e-8

# Linop MATIRF parameters
sigma_x = sigma_y = None
m_gaussian = 64

linop_gauss  = Gaussian2D_MATIRF(
    sigma_x, sigma_y, 
    N1=m_gaussian, N2=m_gaussian
)

product_NREP_Mmeasures = product(r_NREP_p, measures_per_spike)

for idx_test, (rep, m) in tqdm(enumerate(product_NREP_Mmeasures), disable=True):
    print(f'Test number {rep + 1} for m = {m}')
    directory = 'data/'

    name_prefix = f"data_k{N:03d}_m{m:02d}_rep{rep:02d}_"
    name_suffix = ".npz"

    file_name_true_data    = directory + name_prefix + "true"    + name_suffix
    file_name_opCOMP_data  = directory + name_prefix + "opCOMP"  + name_suffix
    file_name_PGD_data     = directory + name_prefix + "PGD"     + name_suffix
    
    seed = rep
    np.random.seed(seed)

    # Creation of positions t and amplitudes a
    t = semi_gridded_init(density=density)
    factor = .8
    shift = linop_gauss.b1 * ((1 - factor) / 2)
    t = t * np.array([factor, factor, 1]) + np.array([shift, shift, 0])
    a = np.random.uniform(low=1, high=2, size=N)

    X_true = np.concatenate((t, a[:, None]), axis=-1)

    # Linop Sketching parameters
    m_sketch = m * N
    linop_sketch = Sketching_Gaussian2D_MATIRF(m_sketch, linop_gauss.sigma_x, linop_gauss.sigma_y)
    
    ####################
    if SAVE_DATA:
        np.savez_compressed(
            file_name_true_data,
            X_true=X_true,
            w=linop_sketch.w,
        )
    ####################

    # Original observation
    y = linop_gauss.Ax(a, t)
    t_grid = np.copy(linop_gauss.grid)
    a_grid = np.copy(y)
    
    # Creation of sketched observation
    y_fourier = linop_sketch.DFT_DiracComb(a_grid, t_grid)

    # Sub-Functions 
    functional = partial(
        g, y=y_fourier, linop=linop_sketch
    )
    gradient_functional = partial(
        gradient_g, y=y_fourier, linop=linop_sketch
    )
    
    #proj = lambda X: X
    proj = partial(
        projection_X,
        eps_proj=0.05,
        cut_off=5e-2
    )
    
    clip = lambda X: X
    
    update_residue = partial(
        FISTA.update_residue_from_y,
        y=y_fourier, linop=linop_sketch
    )
    
    exit_cond = partial(
        FISTA.exit_cond,
        abs_tol_cst=abs_tol_cst, 
        rel_tol_cst=rel_tol_cst,
    )

    # Initialization : OP-COMP
    t_start_opCOMP = perf_counter_ns()
    a_init, t_init, error_opCOMP, residue = opCOMP(
        y=y_fourier, linop=linop_sketch,
        step=step_opCOMP, nb_tests=nb_tests,
        min_iter=N, max_iter=3*N,
        descent_nit=descent_nit_OP_COMP,
        tol_criterion=tol_criterion,
        init_position=init.init_position_random,
        multicore=multicore,
        disable_tqdm_init=disable_tqdm,
    )
    t_finish_opCOMP = perf_counter_ns()
    X_init = np.concatenate((t_init, a_init[:, None]), axis=-1)

    ####################
    if SAVE_DATA:
        np.savez_compressed(
            file_name_opCOMP_data,
            X_init=X_init,
            error_opCOMP=error_opCOMP,
            residue=residue,
            time_opCOMP=np.array([t_finish_opCOMP - t_start_opCOMP])
        )
    ####################

    # Estimation : PGD w/ FISTA Restart
    # X_esti, error_PGD, error_PGD_norm, traj_X, time_diffs = FISTA.FISTA_restart(
    X_esti, error_PGD, error_PGD_norm, time_diffs = FISTA.FISTA_restart(
        X=X_init, nit=descent_nit_PGD, step=step_PGD, 
        functional=functional,
        gradient_functional=gradient_functional,
        exit_cond=exit_cond,
        restart_cond=FISTA.restart_cond,
        project=proj, clip=clip,
        disable_tqdm=disable_tqdm,
    )

    ####################
    if SAVE_DATA:
        np.savez_compressed(
            file_name_PGD_data,
            X_esti=X_esti, error_PGD=error_PGD,
            error_PGD_norm=error_PGD_norm, 
            time_dict=time_diffs, # traj_X=traj_X,
        )
    ####################
    
