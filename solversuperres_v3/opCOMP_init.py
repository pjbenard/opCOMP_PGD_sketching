from os import cpu_count
from multiprocessing import Pool

import numpy as np
import numpy.linalg as npl

from tqdm import tqdm
from functools import partial

from . import FISTA_restart_descent_v2 as FISTA
from . import init_utils as init
from .descent_utilities import clip_domain
from .g_utils import g


def opCOMP(
    y,
    linop,
    step,
    nb_tests=1,
    min_iter=None,
    max_iter=None,
    descent_nit=500,
    init_position=init.init_position_max_val,
    disable_tqdm_FISTA=True,
    disable_tqdm_init=False,
    tol_criterion=0.1,
    multicore=False,
):
    d = linop.d
    bounds_min = linop.bounds["min"]
    bounds_max = linop.bounds["max"]

    norm_y = npl.norm(y)

    r = np.copy(y) # complex
    T = np.array([]).reshape(0, d) # real
    a_best = np.array([]) # real

    errors_criterion = [1]
    errors = [norm_y ** 2]
    # errors.append(npl.norm(r))

    matrix_condition = []
    
    project = lambda X: X
    clip = partial(
        clip_domain,
        linop=linop,
        has_amplitude=False,
    )
    
    descent = partial(
        FISTA.FISTA_restart,
        step=step, nit=descent_nit,
        exit_cond=FISTA.exit_cond,
        restart_cond=FISTA.restart_cond,
        project=project, clip=clip, alpha0=1,
        disable_tqdm=disable_tqdm_FISTA,
    )

    for i in tqdm(range(max_iter), disable=disable_tqdm_init):
        norm_func = partial(
            init.norm_Adeltat_dot_residue,
            residue=r,
            linop=linop
        )
        grad_func = partial(
            init.gradient_Adeltat_dot_residue,
            residue=r,
            linop=linop
        )

        if not multicore:
            ### Sequential
            t_gen = init_position(r, linop)
            t_best, *other_outputs = descent(
                t_gen, functional=norm_func,
                gradient_functional=grad_func
            ) # error_functional, traj_t, time_dict
            r_best = npl.norm(r - linop.Adelta(t_best))
    
            # Tries multiple random initialization to find the best
            for test in range(1, nb_tests):
                t_gen = init_position(r, linop)
                t_int, *other_outputs = descent(
                    t_gen, functional=norm_func,
                    gradient_functional=grad_func
                ) # error_functional, traj_t, time_dict
                r_int = npl.norm(r - linop.Adelta(t_int))
                if r_int < r_best:
                    t_best = np.copy(t_int)
                    r_best = r_int
            ##############
        else:
            ### Multi-proc
            nb_workers = cpu_count() // 2
            descent_residue = partial(
                descent,
                functional=norm_func,
                gradient_functional=grad_func,
            )
    
            global initialize_t
            def initialize_t(idx):
                np.random.seed()
                t_gen = init_position(r, linop)
                t_int, *other_outputs = descent_residue(t_gen)
                r_int = npl.norm(r - linop.Adelta(t_int))
                return (r_int, t_int)
    
            with Pool(processes=nb_workers) as p:
                # Tries multiple random initialization to find the best
                res = p.map(initialize_t, range(nb_tests), chunksize=1)
            r_best, t_best = min(res, key=lambda obj: obj[0])
            ##############

        T = np.concatenate((T, t_best), axis=0)

        M = linop.Adelta(T).T
        a_best = np.real(npl.lstsq(M, y, rcond=None)[0])

        X_best = np.concatenate((T, a_best[:, None]), axis=1)

        r = y - linop.Ax(a_best, T)

        
        error = g(X_best, y, linop)
        errors.append(error)
        error_criterion = (error ** .5) / norm_y
        errors_criterion.append(error_criterion)

        if errors[-1] > errors[-2]:
            a_best = a_best[:-1]
            T = T[:-1]

        if i >= min_iter - 1:
            if errors_criterion[-1] <= tol_criterion:
                break

    a_init = np.copy(a_best)
    t_init = np.copy(T)

    return a_init, t_init, np.array(errors), r
