import numpy as np

from tqdm import tqdm
from typing import Callable, Tuple, Union
from functools import partial
from time import perf_counter_ns

def absolute_tolerance(func_X0, func_X1, tol):
    return func_X1 <= tol

def relative_tolerance(func_X0, func_X1, tol):
    return (abs(func_X1 - func_X0) /  abs(func_X0)) <= tol

def exit_cond(func_X0, func_X1, abs_tol_cst=1e-10, rel_tol_cst=1e-10):
    abs_tol = absolute_tolerance(func_X0, func_X1, tol=abs_tol_cst)
    rel_tol = relative_tolerance(func_X0, func_X1, tol=rel_tol_cst)
    return abs_tol or rel_tol

def restart_cond(X0, X1, G_Y0):
    return np.inner(G_Y0.ravel(), (X1 - X0).ravel()) > 0

def update_residue_from_y(X, y, linop):
    return y - linop.Ax(X[:, -1], X[:, :-1])

def FISTA(
    X: np.ndarray,
    step: float,
    functional: Callable,
    gradient_functional: Callable,
    kit: int,
    exit_cond: Callable = None,
    restart_cond: Callable = None,
    project: Callable = None,
    clip: Callable = None,
    alpha0: float = 1.,
    functional_y:float = None,
) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    
    X0 = X
    Y0 = np.copy(X0)

    error_functional = np.zeros(kit + 1)
    error_functional[0] = functional(X0)
    #### Not univseral
    if functional_y is None:
        functional_y = functional(X0)
    error_functional_norm = np.zeros(kit + 1)
    error_functional_norm[0] = (error_functional[0] / functional_y)**.5
    ##################
    # traj_X = np.zeros((kit + 1, *X0.shape))
    # traj_X[0] = np.copy(X0)
    
    for k in range(1, kit + 1):
        G_Y0 = gradient_functional(Y0)
        X1 = Y0 - step * G_Y0
        X1 = project(X1)
        # X1 = clip(X1)

        error_functional[k] = functional(X1)
        ##################
        error_functional_norm[k] = (functional(X1) / functional_y)**.5
        ##################

        is_projected = np.size(X1) != np.size(X0)
        if is_projected:
            break

        # traj_X[k] = np.copy(X1)

        alpha1 = (1 + (1 + 4 * alpha0**2)**.5) / 2
        beta = (alpha0 - 1) / alpha1
        alpha0 = alpha1
        Y1 = X1 + beta * (X1 - X0)

        if exit_cond(
            error_functional_norm[k - 1],
            error_functional_norm[k]
        ) or restart_cond(X0, X1, G_Y0):
            break

        X0 = np.copy(X1)
        Y0 = np.copy(Y1)
            
    return X1, k, error_functional[1:k+1], error_functional_norm[1:k+1]
    # return X1, k, error_functional[1:k+1], error_functional_norm[1:k+1], traj_X[1:k+1]


def FISTA_restart(
    X: np.ndarray,
    nit: int,
    step: float,
    functional: Callable,
    gradient_functional: Callable,
    exit_cond: Callable = None,
    restart_cond: Callable = None,
    project: Callable = None,
    clip: Callable = None,
    alpha0: float = 1.,
    functional_y: float = None,
    disable_tqdm: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    j = 1
    X0 = np.copy(X)
    error_functional = np.zeros(nit + 1)
    error_functional[0] = functional(X0)
    #### Not univseral
    if functional_y is None:
        functional_y = functional(X0)
    error_functional_norm = np.zeros(nit + 1)
    error_functional_norm[0] = (error_functional[0] / functional_y)**.5
    ####
    # traj_X = np.zeros((nit + 1, *X0.shape))
    # traj_X[0] = np.copy(X0)
    time_dict = [(0, perf_counter_ns())]
    
    with tqdm(total=nit, disable=disable_tqdm) as pbar:
        while j < nit + 1:
            # X1, k, error_func_FISTA, error_func_nom_FISTA, traj_X_FISTA = FISTA(
            X1, k, error_func_FISTA, error_func_nom_FISTA = FISTA(
                X=X0, step=step, kit=nit + 1 - j,
                functional=functional,
                gradient_functional=gradient_functional,
                exit_cond=exit_cond, restart_cond=restart_cond,
                project=project, clip=clip, alpha0=alpha0,
                functional_y=functional_y,
            )
            error_functional[j:j+k] = np.copy(error_func_FISTA)
            error_functional_norm[j:j+k] = np.copy(error_func_nom_FISTA)
            # traj_X[j:j+k, :traj_X_FISTA.shape[1], :] = np.copy(traj_X_FISTA)
            
            pbar.update(k)
            j += k
            time_dict.append((j - 1, perf_counter_ns()))
            
            if (X0.size == X1.size) and exit_cond(
                error_functional_norm[j - 2], 
                error_functional_norm[j - 1]
            ):
                break
                
            X0 = np.copy(X1)

    
    time_diffs_dict = {0: 0}
    time_diffs_dict.update({time_dict[i][0]: time_dict[i][1] - time_dict[i - 1][1] for i in range(1, len(time_dict))})

    return X1, error_functional[:j], error_functional_norm[:j], time_diffs_dict
    # return X1, error_functional[:j], error_functional_norm[:j], traj_X[:j], time_diffs_dict


def FISTA_restart_bcd(
    X: np.ndarray,
    nit: int,
    step: float,
    bcdit_max: int,
    bcd_threshold: float,
    functional: Callable,
    gradient_functional: Callable,
    update_residue: Callable,
    exit_cond: Callable = None,
    restart_cond: Callable = None,
    project: Callable = None,
    clip: Callable = None,
    alpha0: float = 1, 
    bcdit_exponent: float = 1,
    disable_tqdm: bool = False,
) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    
    j = 1
    X0 = np.copy(X)
    error_functional = np.zeros(nit + 1)
    error_functional[0] = functional(X0)
    # traj_X = np.zeros((nit + 1, *X0.shape))
    # traj_X[0] = np.copy(X0)
    fraction_moving = dict()
    norm_gradients = dict()
    time_dict = [(0, perf_counter_ns())]
    
    with tqdm(total=nit, disable=disable_tqdm) as pbar:
        while j < nit + 1:            
            grad_X0 = gradient_functional(X0)
            norm_grad_X0 = np.linalg.norm(grad_X0, axis=-1)
            max_norm_grad_X0 = np.max(norm_grad_X0)
            idx_moving_atoms = norm_grad_X0 > (max_norm_grad_X0 * bcd_threshold)
            idx_frozen_atoms = norm_grad_X0 <= (max_norm_grad_X0 * bcd_threshold)
            # percentile = np.percentile(norm_grad_X0, bcd_threshold)
            # idx_moving_atoms = norm_grad_X0 >= percentile
            # idx_frozen_atoms = norm_grad_X0 < percentile

            norm_gradients[j - 1] = np.copy(norm_grad_X0)
            nb_moving_atoms = np.count_nonzero(idx_moving_atoms)
            fraction_moving[j - 1] = (nb_moving_atoms / idx_moving_atoms.size)

            bcdit = int(fraction_moving[j - 1]**bcdit_exponent * bcdit_max)
            kit = min(nit + 1 - j, bcdit)

            if nb_moving_atoms < X0.shape[0]:
                residue = update_residue(X0[idx_frozen_atoms])
            else:
                residue = update_residue(np.zeros((1, X0.shape[-1])))

            updated_functional = partial(functional, y=residue)
            updated_gradient_functional = partial(gradient_functional, y=residue)
            
            # X1_moving, k, error_func_FISTA, traj_X_FISTA = FISTA(
            X1_moving, k, error_func_FISTA = FISTA(
                X=X0[idx_moving_atoms], step=step, kit=kit, 
                functional=updated_functional,
                gradient_functional=updated_gradient_functional,
                exit_cond=exit_cond, restart_cond=restart_cond,
                project=project, clip=clip, alpha0=alpha0,
            )

            error_functional[j:j+k] = np.copy(error_func_FISTA)
            # traj_X[j:j+k, :traj_X_FISTA.shape[1], :] = np.copy(traj_X_FISTA)

            X1 = np.concatenate((X1_moving, X0[idx_frozen_atoms]))
            
            pbar.update(k)
            j += k
            time_dict.append((j - 1, perf_counter_ns()))
            
            if (X0.size == X1.size) and exit_cond(
                error_functional[j - 2],
                error_functional[j - 1]
            ):
                break
                
            X0 = np.copy(X1)

    time_diffs_dict = {0: 0}
    time_diffs_dict.update({time_dict[i][0]: time_dict[i][1] - time_dict[i - 1][1] for i in range(1, len(time_dict))})

    return X1, error_functional[:j], fraction_moving, norm_gradients, time_diffs_dict
    # return X1, error_functional[:j], traj_X[:j], fraction_moving, norm_gradients, time_diffs_dict