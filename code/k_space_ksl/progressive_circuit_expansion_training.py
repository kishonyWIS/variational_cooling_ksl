"""
Progressive Circuit Expansion Training - New Implementation

This module provides progressive circuit expansion functionality using the new
self-contained KSL model infrastructure (k_space_KSL_model.py) instead of the
old numba-optimized implementation.

The external interface uses the old 6-parameter structure for compatibility:
{'Jx', 'Jy', 'Jz', 'kappa', 'g', 'B'}

Internally, this is converted to the new 9-parameter structure:
{'Jx', 'Jy', 'Jz', 'Delta_A', 'Delta_B', 'g_A', 'g_B', 'B_A', 'B_B'}
"""

import sys
import os
import numpy as np
from scipy.optimize import minimize, basinhopping
import time
import csv
from functools import lru_cache

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add code directory to path for imports
code_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from k_space_KSL_model import (
    create_KSL_hamiltonian, VariationalCircuit, 
    KSLSingleParticleDensityMatrix
)
from time_dependence_functions import get_g, get_B
from interger_chern_number import chern_from_mixed_spdm

# Data directory path (relative to this file)
DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data')

# Set random seed for reproducibility
np.random.seed(42)


# ===== Helper Functions =====

def get_k_grid(n_k_points):
    """Create k-grid from -pi to pi (excluding endpoint) for periodic boundary conditions"""
    return np.linspace(-np.pi, np.pi, n_k_points, endpoint=False)


def create_trotterized_parameters_9param(num_steps, T, g0, B0, B1):
    """
    Create trotterized evolution parameters with 9-parameter structure (new structure).
    
    This is a shared helper function used by both create_trotterized_circuit and
    trotterized_evolution_parameters to avoid code duplication.
    
    Args:
        num_steps: Number of Trotter steps
        T: Total evolution time
        g0, B0, B1: Time-dependent function parameters
    
    Returns:
        Dictionary with 9-parameter structure: ['Jx', 'Jy', 'Jz', 'Delta_A', 'Delta_B', 
        'g_A', 'g_B', 'B_A', 'B_B']. All values are arrays of length num_steps.
        Parameters are time-step multipliers (dt), with g and B being time-dependent.
    """
    # Time step for trotterization
    dt = T / num_steps
    
    # Initialize parameters - all terms get dt per step (magnitudes are in Hamiltonian term strengths)
    # For g and B, we'll multiply by the time-dependent values
    parameters = {
        'Jx': np.ones(num_steps) * dt,
        'Jy': np.ones(num_steps) * dt,
        'Jz': np.ones(num_steps) * dt,
        'Delta_A': np.ones(num_steps) * dt,
        'Delta_B': np.ones(num_steps) * dt,
        'g_A': np.ones(num_steps),
        'g_B': np.ones(num_steps),
        'B_A': np.ones(num_steps),
        'B_B': np.ones(num_steps),
    }
    
    # Apply time-dependent scaling for g and B terms
    for step in range(num_steps):
        t = step * dt
        g_t = get_g(t, g0, T, T/4)
        B_t = get_B(t, B0, B1, T)
        
        # g and B parameters are the time-dependent values times dt
        parameters['g_A'][step] = g_t * dt
        parameters['g_B'][step] = g_t * dt
        parameters['B_A'][step] = B_t * dt
        parameters['B_B'][step] = B_t * dt
    
    return parameters


# ===== Parameter Conversion Functions =====

def convert_parameters_to_new_structure(parameters):
    """
    Convert old 6-parameter structure to new 9-parameter structure.
    
    Args:
        parameters: Dictionary with keys ['Jx', 'Jy', 'Jz', 'kappa', 'g', 'B']
                   Each value is an array of length p (number of layers) containing time-step multipliers (dt)
    
    Returns:
        Dictionary with keys ['Jx', 'Jy', 'Jz', 'Delta_A', 'Delta_B', 'g_A', 'g_B', 'B_A', 'B_B']
        Each value is an array of length p
    
    Note:
        Parameters are now just time-step multipliers (dt). The magnitudes (Jx, Jy, Jz, kappa, Delta)
        are already in the Hamiltonian term strengths. Parameters are passed through directly.
    """
    # Create new parameter structure - parameters are just time-step multipliers
    # All magnitudes are already in the Hamiltonian term strengths
    new_parameters = {
        'Jx': parameters['Jx'].copy(),
        'Jy': parameters['Jy'].copy(),
        'Jz': parameters['Jz'].copy(),
        'Delta_A': parameters['kappa'].copy(),  # Just dt, Delta magnitude is in Hamiltonian
        'Delta_B': parameters['kappa'].copy(),    # Same for both
        'g_A': parameters['g'].copy(),
        'g_B': parameters['g'].copy(),
        'B_A': parameters['B'].copy(),
        'B_B': parameters['B'].copy(),
    }
    
    return new_parameters


# ===== Simulation Functions =====

@lru_cache(maxsize=None)
def _get_cached_hamiltonian_data(kx, ky, Jx, Jy, Jz, kappa):
    """
    Cached helper function to compute Hamiltonian and ground state data.
    
    This function caches expensive computations (Hamiltonian creation and ground state)
    that don't depend on variational parameters, only on (kx, ky, Jx, Jy, Jz, kappa).
    
    Args:
        kx, ky: Momentum values
        Jx, Jy, Jz, kappa: Kitaev parameters
    
    Returns:
        tuple: (system_hamiltonian, ground_state_matrix, E_gs, circuit_hamiltonian)
    """
    # Create system Hamiltonian (for energy measurement and ground state)
    system_hamiltonian = create_KSL_hamiltonian(kx, ky, Jx=Jx, Jy=Jy, Jz=Jz, kappa=kappa, g=0.0, B=0.0)
    
    # Get ground state
    ground_state_matrix = system_hamiltonian.get_ground_state()
    E_gs = system_hamiltonian.compute_energy(ground_state_matrix)
    
    # Create circuit Hamiltonian (g and B set to 1.0 to ensure terms exist)
    circuit_hamiltonian = create_KSL_hamiltonian(kx, ky, Jx=Jx, Jy=Jy, Jz=Jz, kappa=kappa, g=1.0, B=1.0)
    
    return system_hamiltonian, ground_state_matrix, E_gs, circuit_hamiltonian


def simulate_single_kpoint(kx, ky, parameters, n_cycles, Jx, Jy, Jz, kappa):
    """
    Simulate variational circuit for a single k-point.
    
    Args:
        kx, ky: Momentum values
        parameters: Dictionary with old 6-parameter structure
        n_cycles: Number of cooling cycles to perform
        Jx, Jy, Jz, kappa: Kitaev parameters
    
    Returns:
        tuple: (final_state, E_diff_list, E_gs, single_particle_dm)
            - final_state: KSLSingleParticleDensityMatrix after all cycles
            - E_diff_list: List of energy differences after each cycle
            - E_gs: Ground state energy
            - single_particle_dm: 6x6 density matrix
    """
    # Get cached Hamiltonian and ground state data (independent of parameters)
    system_hamiltonian, ground_state_matrix, E_gs, circuit_hamiltonian = _get_cached_hamiltonian_data(
        kx, ky, Jx, Jy, Jz, kappa
    )
    
    # Convert parameters to new structure
    new_parameters = convert_parameters_to_new_structure(parameters)
    
    # Create variational circuit
    circuit = VariationalCircuit(circuit_hamiltonian, new_parameters)
    Ud = circuit.get_unitary()
    
    # Initialize state
    state = KSLSingleParticleDensityMatrix(system_hamiltonian.system_size)
    state.initialize('random')
    
    # Store energy differences after each cycle
    E_diff_list = []
    
    # Perform cooling cycles
    for cycle in range(n_cycles):

        state.reset_all_tau()
        
        # Apply the variational circuit
        state.evolve_state_with_unitary(Ud)
        
        # Calculate energy after this cycle
        E_current = system_hamiltonian.compute_energy(state.matrix)
        E_diff = E_current - E_gs
        E_diff_list.append(E_diff)
    
    return state, E_diff_list, E_gs, state.matrix


def simulate_grid(kx_list, ky_list, parameters, n_cycles, Jx, Jy, Jz, kappa, verbose=False):
    """
    Simulate variational circuit over a momentum grid.
    
    Args:
        kx_list, ky_list: Momentum space grids
        parameters: Dictionary with old 6-parameter structure
        n_cycles: Number of cooling cycles to perform
        Jx, Jy, Jz, kappa: Kitaev parameters
        verbose: Whether to print progress
    
    Returns:
        tuple: (E_diff, single_particle_dm)
            - E_diff: Array of shape (n_cycles, n_kx, n_ky) with energy differences
            - single_particle_dm: Array of shape (n_kx, n_ky, 6, 6) with density matrices
    """
    n_kx = len(kx_list)
    n_ky = len(ky_list)
    
    # Initialize output arrays
    E_diff = np.zeros((n_cycles, n_kx, n_ky), dtype=float)
    single_particle_dm = np.zeros((n_kx, n_ky, 6, 6), dtype=complex)
    
    total_points = n_kx * n_ky
    point_count = 0
    
    for i_kx, kx in enumerate(kx_list):
        for i_ky, ky in enumerate(ky_list):
            if verbose and point_count % 10 == 0:
                print(f"  Processing k-point {point_count+1}/{total_points}: kx={kx:.4f}, ky={ky:.4f}")
            
            # Simulate this k-point
            _, E_diff_list, _, dm = simulate_single_kpoint(kx, ky, parameters, n_cycles, Jx, Jy, Jz, kappa)
            
            # Store results
            for cycle in range(n_cycles):
                E_diff[cycle, i_kx, i_ky] = E_diff_list[cycle]
            single_particle_dm[i_kx, i_ky, :, :] = dm
            
            point_count += 1
    
    return E_diff, single_particle_dm


def simulate_grid_with_analysis(kx_list, ky_list, parameters, n_cycles, Jx, Jy, Jz, kappa):
    """
    Simulate variational circuit over a momentum grid and calculate analysis metrics.
    
    Args:
        kx_list, ky_list: Momentum space grids
        parameters: Dictionary with old 6-parameter structure
        n_cycles: Number of cooling cycles to perform
        Jx, Jy, Jz, kappa: Kitaev parameters
    
    Returns:
        tuple: (E_diff, single_particle_dm, total_chern_number, system_chern_number, 
                bath_chern_number, energy_density)
    """
    # Run simulation
    E_diff, single_particle_dm = simulate_grid(kx_list, ky_list, parameters, n_cycles, 
                                                Jx, Jy, Jz, kappa, verbose=False)
    
    # Calculate Chern numbers
    total_chern_number = chern_from_mixed_spdm(single_particle_dm)
    system_chern_number = chern_from_mixed_spdm(single_particle_dm[:, :, :2, :2])
    bath_chern_number = chern_from_mixed_spdm(single_particle_dm[:, :, 2:, 2:])
    
    # Calculate average energy density (use final cycle if multiple cycles)
    if n_cycles == 1:
        energy_density = np.nanmean(E_diff) / 2  # E_diff is 2D
    else:
        energy_density = np.nanmean(E_diff[-1, :, :]) / 2  # E_diff is 3D, use final cycle
    
    return E_diff, single_particle_dm, total_chern_number, system_chern_number, bath_chern_number, energy_density


# ===== Parameter Management Functions =====

def trotterized_evolution_parameters(num_steps, T, g0, B0, B1):
    """
    Create trotterized evolution parameters based on the original adiabatic evolution.
    This provides a starting point for the variational optimization.
    
    Uses the shared create_trotterized_parameters_9param function and converts
    to the old 6-parameter structure.
    
    Note: Parameters are now just time-step multipliers (dt). The magnitudes (Jx, Jy, Jz, kappa)
    are already in the Hamiltonian term strengths.
    
    Args:
        num_steps: Number of Trotter steps (p)
        T: Total evolution time
        g0, B0, B1: Time-dependent function parameters
    
    Returns:
        Dictionary with old 6-parameter structure (all values are real time-step multipliers)
    """
    # Get 9-parameter structure from shared function
    params_9 = create_trotterized_parameters_9param(num_steps, T, g0, B0, B1)
    
    # Convert to old 6-parameter structure
    parameters = {
        'Jx': params_9['Jx'].copy(),
        'Jy': params_9['Jy'].copy(),
        'Jz': params_9['Jz'].copy(),
        'kappa': params_9['Delta_A'].copy(),  # Delta_A and Delta_B are the same, use one
        'g': params_9['g_A'].copy(),  # g_A and g_B are the same, use one
        'B': params_9['B_A'].copy()   # B_A and B_B are the same, use one
    }
    
    return parameters


def expand_strength_durations(old_parameters, old_p, new_p):
    """
    Expand parameters from old_p layers to new_p layers by inserting zero layers.
    
    Args:
        old_parameters: Dictionary with old 6-parameter structure, each array of length old_p
        old_p: Current number of layers
        new_p: Target number of layers
    
    Returns:
        Dictionary with same structure, each array of length new_p
    """
    if new_p <= old_p:
        raise ValueError(f"new_p ({new_p}) must be greater than old_p ({old_p})")
    
    new_parameters = {}
    for key in old_parameters:
        old_array = old_parameters[key]
        # Insert zeros in the middle
        # Strategy: keep first half, insert zeros, keep second half
        mid_point = old_p // 2
        new_array = np.zeros(new_p, dtype=old_array.dtype)
        new_array[:mid_point] = old_array[:mid_point]
        new_array[new_p - (old_p - mid_point):] = old_array[mid_point:]
        new_parameters[key] = new_array
    
    return new_parameters


def strength_durations_to_vector(parameters, p):
    """
    Convert parameters dictionary to a flat vector for optimization.
    
    Args:
        parameters: Dictionary with old 6-parameter structure
        p: Number of layers
    
    Returns:
        Flat numpy array of length 6 * p
    """
    vector = []
    for term in ['Jx', 'Jy', 'Jz', 'kappa', 'g', 'B']:
        if len(parameters[term]) != p:
            raise ValueError(f"Parameter '{term}' has length {len(parameters[term])}, expected {p}")
        vector.extend(parameters[term])
    return np.array(vector)


def vector_to_strength_durations(vector, p):
    """
    Convert flat vector back to parameters dictionary.
    
    Args:
        vector: Flat numpy array of length 6 * p
        p: Number of layers
    
    Returns:
        Dictionary with old 6-parameter structure
    """
    if len(vector) != 6 * p:
        raise ValueError(f"Vector length {len(vector)} doesn't match expected 6 * p = {6 * p}")
    
    parameters = {}
    start_idx = 0
    for term in ['Jx', 'Jy', 'Jz', 'kappa', 'g', 'B']:
        end_idx = start_idx + p
        parameters[term] = vector[start_idx:end_idx]
        start_idx = end_idx
    
    return parameters


# ===== Optimization Functions =====

def objective_function(vector, kx_list, ky_list, n_cycles, p, Jx, Jy, Jz, kappa, verbose=False):
    """
    Objective function to minimize: RMS energy density.
    
    Args:
        vector: Optimization parameters (flat vector)
        kx_list, ky_list: Momentum space grids
        n_cycles: Number of cooling cycles to perform
        p: Number of layers
        Jx, Jy, Jz, kappa: Kitaev parameters
        verbose: Whether to print progress
    
    Returns:
        RMS energy density
    """
    # Convert vector back to parameters
    parameters = vector_to_strength_durations(vector, p)
    
    if verbose:
        print(f"    Evaluating objective function on {len(kx_list)}x{len(ky_list)} grid with {n_cycles} cycles...")
    
    # Run simulation
    E_diff, _ = simulate_grid(kx_list, ky_list, parameters, n_cycles, Jx, Jy, Jz, kappa, verbose)
    
    # Calculate RMS energy density
    if n_cycles == 1:
        rms_energy_density = np.sqrt(np.nanmean(E_diff**2)) / 2
    else:
        rms_energy_density = np.sqrt(np.nanmean(E_diff[-1, :, :]**2)) / 2
    
    if verbose:
        print(f"    RMS energy density: {rms_energy_density:.6f}")
    
    return rms_energy_density


def optimize_strength_durations(kx_list, ky_list, n_cycles, p, initial_parameters, 
                                method='L-BFGS-B', Jx=1.0, Jy=1.0, Jz=1.0, kappa=1.0, epochs=1000):
    """
    Optimize parameters to minimize energy density (local optimization).
    
    Args:
        kx_list, ky_list: Momentum space grid
        n_cycles: Number of cooling cycles to perform
        p: Number of layers
        initial_parameters: Initial guess (old 6-parameter structure)
        method: Optimization method ('L-BFGS-B', 'SLSQP', etc.)
        Jx, Jy, Jz, kappa: Kitaev parameters
        epochs: Maximum number of iterations
    
    Returns:
        tuple: (optimized_parameters, optimization_result)
    """
    print("Starting local optimization...")
    
    if initial_parameters is None:
        raise ValueError("initial_parameters cannot be None for local optimization")
    
    # Convert to vector
    initial_vector = strength_durations_to_vector(initial_parameters, p)
    
    bounds = None
    
    print(f"Optimizing {len(initial_vector)} parameters using {method}")
    print(f"Initial energy density: {objective_function(initial_vector, kx_list, ky_list, n_cycles, p, Jx, Jy, Jz, kappa, verbose=True):.6f}")
    
    # Set up optimization options
    options = {'maxiter': epochs, 'disp': True, 'ftol': 1e-9, 'gtol': 1e-5}
    max_iter = options['maxiter']
    
    # Create a wrapper class to track function evaluations efficiently
    class ObjectiveWrapper:
        def __init__(self, func, kx_list, ky_list, n_cycles, p, Jx, Jy, Jz, kappa):
            self.func = func
            self.kx_list = kx_list
            self.ky_list = ky_list
            self.n_cycles = n_cycles
            self.p = p
            self.Jx = Jx
            self.Jy = Jy
            self.Jz = Jz
            self.kappa = kappa
            self.last_x = None
            self.last_value = None
            self.eval_count = 0
            
        def __call__(self, x):
            self.eval_count += 1
            self.last_x = x.copy()
            self.last_value = self.func(x, self.kx_list, self.ky_list, self.n_cycles, 
                                       self.p, self.Jx, self.Jy, self.Jz, self.kappa, verbose=False)
            return self.last_value
    
    # Create wrapper and callback
    obj_wrapper = ObjectiveWrapper(objective_function, kx_list, ky_list, n_cycles, p, Jx, Jy, Jz, kappa)
    iteration_count = [0]
    iteration_start_time = [time.time()]
    
    def callback(xk):
        iteration_count[0] += 1
        current_time = time.time()
        iteration_time = current_time - iteration_start_time[0]
        
        # Use the cached value from the wrapper if available
        if obj_wrapper.last_value is not None and np.allclose(xk, obj_wrapper.last_x):
            current_energy = obj_wrapper.last_value
        else:
            # Fallback: compute if not available
            current_energy = objective_function(xk, kx_list, ky_list, n_cycles, p, Jx, Jy, Jz, kappa, verbose=False)
        
        print(f"  Iteration {iteration_count[0]}/{max_iter}: Energy density = {current_energy:.6f} (Time: {iteration_time:.2f}s)")
        iteration_start_time[0] = current_time
        return False
    
    # Perform optimization
    result = minimize(
        obj_wrapper,
        initial_vector,
        method=method,
        bounds=bounds,
        callback=callback,
        options=options
    )
    
    # Convert result back to parameters
    optimized_parameters = vector_to_strength_durations(result.x, p)
    
    print(f"Optimization completed!")
    print(f"Final energy density: {result.fun:.6f}")
    print(f"Success: {result.success}")
    print(f"Iterations: {result.nit}")
    print(f"Function evaluations: {result.nfev}")
    print(f"Gradient evaluations: {result.njev}")
    print(f"Total objective function calls: {obj_wrapper.eval_count}")
    
    return optimized_parameters, result


def optimize_strength_durations_global(kx_list, ky_list, n_cycles, p, niter=50, 
                                       Jx=1.0, Jy=1.0, Jz=1.0, kappa=1.0, T=50.0, g0=0.5, B0=7.0, B1=0.0):
    """
    Optimize parameters to minimize energy density (global optimization using basin hopping).
    
    Args:
        kx_list, ky_list: Momentum space grid
        n_cycles: Number of cooling cycles to perform
        p: Number of layers
        niter: Number of basin hopping iterations
        Jx, Jy, Jz, kappa: Kitaev parameters
        T, g0, B0, B1: Parameters for trotterized initialization
    
    Returns:
        tuple: (optimized_parameters, optimization_result)
    """
    print("Starting global optimization...")
    
    # Start from trotterized parameters
    initial_parameters = trotterized_evolution_parameters(p, T, g0, B0, B1)
    initial_vector = strength_durations_to_vector(initial_parameters, p)
    
    print(f"Optimizing {len(initial_vector)} parameters using basin hopping")
    initial_energy = objective_function(initial_vector, kx_list, ky_list, n_cycles, p, Jx, Jy, Jz, kappa, verbose=False)
    print(f"Initial energy density: {initial_energy:.6f}")
    
    # Set up optimization options
    minimizer_kwargs = {
        'method': 'L-BFGS-B',
        'options': {'maxiter': 100, 'ftol': 1e-9, 'gtol': 1e-5}
    }
    
    # Create wrapper for objective function
    def objective_wrapper(x):
        return objective_function(x, kx_list, ky_list, n_cycles, p, Jx, Jy, Jz, kappa, verbose=False)
    
    # Perform basin hopping
    result = basinhopping(
        objective_wrapper,
        initial_vector,
        niter=niter,
        minimizer_kwargs=minimizer_kwargs,
        stepsize=0.5,
        T=1.0
    )
    
    # Convert result back to parameters
    optimized_parameters = vector_to_strength_durations(result.x, p)
    
    print(f"Global optimization completed!")
    print(f"Final energy density: {result.fun:.6f}")
    print(f"Function evaluations: {result.nfev}")
    
    return optimized_parameters, result


# ===== Experiment Functions =====

def run_single_experiment(p_val, kx_list_train, ky_list_train, kx_list_test, ky_list_test,
                         initial_parameters=None, use_global=False, Jx=1.0, Jy=1.0, Jz=1.0, kappa=1.0,
                         n_cycles_train=5, n_cycles_test=40, T=50.0, g0=0.5, B0=7.0, B1=0.0, epochs=1000,
                         global_niter=50):
    """
    Run a single experiment with given parameters.
    
    Args:
        p_val: Number of layers in the circuit
        kx_list_train, ky_list_train: Training momentum space grids
        kx_list_test, ky_list_test: Test momentum space grids
        initial_parameters: Optional initial parameters (if None, uses trotterized)
        use_global: If True, use global optimization; if False, use local optimization
        Jx, Jy, Jz, kappa: Kitaev parameters
        n_cycles_train: Number of cooling cycles for training
        n_cycles_test: Number of cooling cycles for testing
        T, g0, B0, B1: Parameters for trotterized initialization
        epochs: Maximum iterations for local optimization
        global_niter: Number of basin hopping iterations for global optimization
    
    Returns:
        tuple: (energy_density_train, energy_density_test, optimized_parameters)
    """
    if use_global:
        # Use global optimization
        print(f"Using global optimization for p={p_val}")
        optimized_parameters, opt_result = optimize_strength_durations_global(
            kx_list_train, ky_list_train[ky_list_train >= 0], 
            n_cycles=n_cycles_train,
            p=p_val,
            niter=global_niter,
            Jx=Jx, Jy=Jy, Jz=Jz, kappa=kappa,
            T=T, g0=g0, B0=B0, B1=B1
        )
    else:
        # Use local optimization
        # Get initialization parameters
        if initial_parameters is None:
            initial_parameters = trotterized_evolution_parameters(p_val, T, g0, B0, B1)
        
        # Optimize the parameters on training data
        optimized_parameters, opt_result = optimize_strength_durations(
            kx_list_train, ky_list_train[ky_list_train >= 0], 
            n_cycles=n_cycles_train,
            p=p_val,
            initial_parameters=initial_parameters,
            method='L-BFGS-B',
            Jx=Jx, Jy=Jy, Jz=Jz, kappa=kappa,
            epochs=epochs
        )
    
    # Evaluate on training grid (using full grid and n_cycles_test)
    E_diff_train, _ = simulate_grid(kx_list_train, ky_list_train, optimized_parameters, 
                                     n_cycles=n_cycles_test, Jx=Jx, Jy=Jy, Jz=Jz, kappa=kappa, verbose=False)
    if n_cycles_test == 1:
        energy_density_train = np.nanmean(E_diff_train) / 2
    else:
        energy_density_train = np.nanmean(E_diff_train[-1, :, :]) / 2
    
    # Evaluate on test grid (using n_cycles_test)
    E_diff_test, _ = simulate_grid(kx_list_test, ky_list_test, optimized_parameters, 
                                    n_cycles=n_cycles_test, Jx=Jx, Jy=Jy, Jz=Jz, kappa=kappa, verbose=False)
    if n_cycles_test == 1:
        energy_density_test = np.nanmean(E_diff_test) / 2
    else:
        energy_density_test = np.nanmean(E_diff_test[-1, :, :]) / 2
    
    return energy_density_train, energy_density_test, optimized_parameters


# ===== Parameter Save/Load Functions =====

def save_optimized_parameters_for_res_p(parameters, res, p, output_dir=None):
    """
    Save optimized parameters to file.
    
    Args:
        parameters: Dictionary with old 6-parameter structure
        res: Resolution value
        p: Number of layers
        output_dir: Directory to save parameters (default: DATA_DIR/optimized_parameters)
    
    Returns:
        Filename of saved file
    """
    if output_dir is None:
        output_dir = os.path.join(DATA_DIR, 'optimized_parameters')
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as numpy file
    filename = os.path.join(output_dir, f'res_{res}_p_{p}.npz')
    np.savez(filename, **parameters)
    
    return filename


def load_optimized_parameters_for_res_p(res, p, input_dir=None):
    """
    Load optimized parameters from file.
    
    Args:
        res: Resolution value
        p: Number of layers
        input_dir: Directory containing saved parameters (default: DATA_DIR/optimized_parameters)
    
    Returns:
        Dictionary with old 6-parameter structure, or None if file doesn't exist
    """
    if input_dir is None:
        input_dir = os.path.join(DATA_DIR, 'optimized_parameters')
    
    filename = os.path.join(input_dir, f'res_{res}_p_{p}.npz')
    
    if not os.path.exists(filename):
        return None
    
    data = np.load(filename)
    parameters = {key: data[key] for key in data.keys()}
    return parameters


def save_results_to_csv(results, filename):
    """
    Save results to CSV file.
    
    Args:
        results: List of dictionaries with results
        filename: Output CSV filename
    """
    if not results:
        return
    
    fieldnames = ['res', 'n_k_points_train', 'p', 'energy_density_train', 'energy_density_test']
    
    # Check if error field exists in any result
    if any('error' in r for r in results):
        fieldnames.append('error')
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)


# ===== Main Progressive Circuit Expansion Function =====

def run_progressive_circuit_expansion(output_csv=None, params_output_dir=None,
                                     Jx=1.0, Jy=1.0, Jz=1.0, kappa=1.0,
                                     n_cycles_train=5, n_cycles_test=40, T=50.0, g0=0.5, B0=7.0, B1=0.0, epochs=1000,
                                     global_niter=50):
    """
    Run progressive circuit expansion over res and p values.
    
    Args:
        output_csv: Filename for saving optimized results
        params_output_dir: Directory to save optimized parameters for each (res, p) combination
        Jx, Jy, Jz, kappa: Kitaev parameters
        n_cycles_train: Number of cooling cycles for training
        n_cycles_test: Number of cooling cycles for testing
        T, g0, B0, B1: Parameters for trotterized initialization
        epochs: Maximum iterations for local optimization
        global_niter: Number of basin hopping iterations for global optimization (default 10 for faster testing)
    
    Returns:
        list: optimized_results as a list of dictionaries
    """
    if output_csv is None:
        output_csv = os.path.join(DATA_DIR, 'progressive_circuit_expansion_results.csv')
    if params_output_dir is None:
        params_output_dir = os.path.join(DATA_DIR, 'optimized_parameters')
    
    # Parameter ranges
    res_values = [1, 2, 3, 4]
    p_values = [2, 3, 4, 5, 6, 7]
    
    # Get the smallest p value (will use global optimization for this)
    smallest_p = min(p_values)
    
    # Fixed test grid size (use current default)
    n_k_points_test_fixed = 6 * 20  # 120
    
    # Store results
    results = []
    
    total_experiments = len(res_values) * len(p_values)
    experiment_count = 0
    
    print(f"\n{'='*60}")
    print("PROGRESSIVE CIRCUIT EXPANSION EXPERIMENT")
    print(f"{'='*60}")
    print(f"Testing {len(res_values)} res values: {res_values}")
    print(f"Testing {len(p_values)} p values: {p_values}")
    print(f"Smallest p={smallest_p} will use GLOBAL optimization")
    print(f"Larger p values will use LOCAL optimization")
    print(f"Total experiments: {total_experiments}")
    print(f"Test grid size: {n_k_points_test_fixed}x{n_k_points_test_fixed}")
    print(f"Training cycles: {n_cycles_train}, Test cycles: {n_cycles_test}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # Track previous optimized parameters for each res value
    # Each res value is completely independent - starts fresh with global optimization for smallest p
    previous_optimized = {}  # key: res, value: (prev_p, optimized_parameters)
    
    kx_list_test = get_k_grid(n_k_points_test_fixed)
    ky_list_test = get_k_grid(n_k_points_test_fixed)
    
    for res in res_values:
        # Each res starts fresh - no state carried over from previous res values
        # Generate grids for this res value (they're the same for all p values)
        n_k_points_train_res = 6 * res
        kx_list_train = get_k_grid(n_k_points_train_res)
        ky_list_train = get_k_grid(n_k_points_train_res)
        
        for p_val in p_values:
            experiment_count += 1
            exp_start_time = time.time()
            
            print(f"\n[{experiment_count}/{total_experiments}] Running res={res}, p={p_val}...")
            
            # Now run optimization
            try:
                # Determine optimization strategy
                # For the smallest p value, use global optimization
                # For larger p values, expand from previous optimized circuit and use local optimization
                if p_val == smallest_p:
                    # Smallest p: use global optimization (no initial guess needed)
                    print(f"  Using GLOBAL optimization for smallest p={p_val}")
                    energy_density_train, energy_density_test, optimized_parameters = run_single_experiment(
                        p_val, kx_list_train, ky_list_train, kx_list_test, ky_list_test, 
                        use_global=True,
                        Jx=Jx, Jy=Jy, Jz=Jz, kappa=kappa,
                        n_cycles_train=n_cycles_train, n_cycles_test=n_cycles_test,
                        T=T, g0=g0, B0=B0, B1=B1, epochs=epochs,
                        global_niter=global_niter
                    )
                elif res in previous_optimized:
                    prev_p, prev_optimized = previous_optimized[res]
                    if p_val > prev_p:
                        # Expand previous optimized circuit by inserting zero layers
                        print(f"  Expanding circuit from p={prev_p} to p={p_val} and using LOCAL optimization...")
                        initial_parameters = expand_strength_durations(prev_optimized, prev_p, p_val)
                        energy_density_train, energy_density_test, optimized_parameters = run_single_experiment(
                            p_val, kx_list_train, ky_list_train, kx_list_test, ky_list_test, 
                            initial_parameters=initial_parameters, use_global=False,
                            Jx=Jx, Jy=Jy, Jz=Jz, kappa=kappa,
                            n_cycles_train=n_cycles_train, n_cycles_test=n_cycles_test,
                            T=T, g0=g0, B0=B0, B1=B1, epochs=epochs
                        )
                    else:
                        # p decreased (shouldn't happen), restart with global optimization
                        raise ValueError(f"p decreased from {prev_p} to {p_val}, this shouldn't happen")
                else:
                    # This shouldn't happen if we handle smallest_p correctly above
                    raise ValueError(f"No previous optimization found for res={res}, p={p_val}, and p != smallest_p")
                
                # Store optimized parameters for next iteration
                previous_optimized[res] = (p_val, optimized_parameters)
                
                # Save optimized parameters to file
                param_filename = save_optimized_parameters_for_res_p(
                    optimized_parameters, res, p_val, params_output_dir
                )
                
                results.append({
                    'res': res,
                    'n_k_points_train': 6 * res,
                    'p': p_val,
                    'energy_density_train': energy_density_train,
                    'energy_density_test': energy_density_test
                })
                
                exp_time = time.time() - exp_start_time
                print(f"  Completed in {exp_time:.1f}s")
                
                # Save results after each experiment
                save_results_to_csv(results, output_csv)
                print(f"  Results saved to {output_csv}")
                print(f"  Parameters saved to {param_filename}")
                
            except Exception as e:
                print(f"  ✗ Failed: {str(e)}")
                import traceback
                traceback.print_exc()
                results.append({
                    'res': res,
                    'n_k_points_train': 6 * res,
                    'p': p_val,
                    'energy_density_train': np.nan,
                    'energy_density_test': np.nan,
                    'error': str(e)
                })
                # On error, don't update previous_optimized - will restart with trotterized next time
                # Still save the error result
                save_results_to_csv(results, output_csv)
                print(f"  Results saved to {output_csv} (with error)")
    
    total_time = time.time() - start_time
    
    # Final save
    save_results_to_csv(results, output_csv)
    
    print(f"\n{'='*60}")
    print("PROGRESSIVE CIRCUIT EXPANSION COMPLETE")
    print(f"{'='*60}")
    print(f"Total experiments: {experiment_count}/{total_experiments}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average time per experiment: {total_time/total_experiments:.1f} seconds")
    print(f"Optimized results saved to: {output_csv}")
    print(f"{'='*60}\n")
    
    return results


if __name__ == "__main__":
    import sys
    run_progressive_circuit_expansion(output_csv=os.path.join(DATA_DIR, 'progressive_circuit_expansion_results.csv'))

