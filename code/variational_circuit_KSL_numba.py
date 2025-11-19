from itertools import product
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize, basinhopping
import numba
from numba import jit, complex128, float64, int32
import time
import csv
import sys
import os
from interger_chern_number import chern_from_mixed_spdm, get_chern_number_from_single_particle_dm

# Add project root to path to access root-level dependencies
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from time_dependence_functions import get_g, get_B
from translational_invariant_KSL import get_KSL_model, get_Delta, get_f

# Data directory path (relative to this file)
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')

# Set random seed for reproducibility
np.random.seed(42)

# Pauli matrices - defined as constants for Numba
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

# Parameters
kappa = 1.
Jx = 1.
Jy = 1.
Jz = 1.

# trotterization parameters
g0 = 0.5
B1 = 0.
B0 = 7.
T = 50
smoothed_g = lambda tt: get_g(tt, g0, T, T/4) #lambda tt: 0#
smoothed_B = lambda tt: get_B(tt, B0, B1, T) #lambda tt: 0#

# Variational circuit parameters
p = 5  # Number of layers in the variational circuit
n_k_points_train = 6*3  # Training grid size
n_k_points_test = 6*20   # Testing grid size

# Cooling cycle parameters
n_cycles_train = 5    # Number of cooling cycles for training
n_cycles_test = 10     # Number of cooling cycles for testing
epochs = 1000


# ===== Core Circuit Functions (Numba-optimized) =====

@jit(nopython=True, cache=True)
def pauli_exponentiation_numba(a_n):
    """
    Numba-optimized Pauli exponentiation
    Compute exp(i * a * (n_vec · σ)) using the formula:
    exp(i * a * (n_vec · σ)) = I * cos(a) + i * (n_vec · σ) * sin(a)
    
    Args:
        a_n: unormalized vector a*n_vec (3-element array)
    
    Returns:
        2x2 unitary matrix
    """
    # Calculate norm
    a = np.sqrt(a_n[0]**2 + a_n[1]**2 + a_n[2]**2)
    if a < 1e-10:
        return np.eye(2, dtype=np.complex128)
    
    n_vec = a_n / a
    
    # Precompute trigonometric values
    cos_a = np.cos(a)
    sin_a = np.sin(a)
    
    # Identity matrix
    I = np.eye(2, dtype=np.complex128)
    
    # Construct n_vec · σ
    n_dot_sigma = n_vec[0] * sigma_x + n_vec[1] * sigma_y + n_vec[2] * sigma_z
    
    # Apply the formula
    return I * cos_a + 1j * n_dot_sigma * sin_a


@jit(nopython=True, cache=True)
def matrix_multiply_6x6(A, B):
    """
    Optimized 6x6 matrix multiplication for Numba
    """
    C = np.zeros((6, 6), dtype=np.complex128)
    for i in range(6):
        for j in range(6):
            for k in range(6):
                C[i, j] += A[i, k] * B[k, j]
    return C


@jit(nopython=True, cache=True)
def create_variational_circuit_numba(strength_durations_array, kx, ky, num_layers):
    """
    Numba-optimized variational circuit creation
    
    Args:
        strength_durations_array: 2D array of shape (6, num_layers) where:
            - Row 0: Jx values (unitless, periodic mod 2π, stored as 2×actual, used as actual/2)
            - Row 1: Jy values (unitless, periodic mod 2π, stored as 2×actual, used as actual/2)
            - Row 2: Jz values (unitless, periodic mod 2π, stored as 2×actual, used as actual/2)
            - Row 3: kappa values (unitless, periodic mod 2π, stored as 2×actual, used as actual/2)
            - Row 4: g values (unitless, periodic mod 2π, stored as 2×actual, used as actual/2)
            - Row 5: B values (unitless, periodic mod 2π, stored as 2×actual, used as actual/2)
        kx, ky: momentum values
        num_layers: number of layers in the circuit
    
    Returns:
        Ud: 6x6 unitary matrix representing the variational circuit
    """
    Ud = np.eye(6, dtype=np.complex128)
    
    # Precompute trigonometric values for this k-point
    cos_kx = np.cos(kx)
    sin_kx = np.sin(kx)
    cos_ky = np.cos(ky)
    sin_ky = np.sin(ky)
    
    # Compute Delta_without_kappa = 4*(sin(kx) - sin(ky) + sin(ky-kx)) (kappa-independent)
    Delta_without_kappa = 4.0 * (sin_kx - sin_ky + np.sin(ky - kx))
    
    # Apply num_layers layers
    # All parameters are stored as 2× their actual value (periodic mod 2π)
    # All parameters are divided by 2 when used in the circuit
    for layer in range(num_layers):
        # Jx term (trainable, unitless parameter, periodic mod 2π, stored as 2×actual, use parameter/2 in circuit)
        strength_duration_val = strength_durations_array[0, layer]
        a_n = np.array([0.0, -2.0 * (strength_duration_val / 2.0), 0.0])
        U_term = pauli_exponentiation_numba(a_n)
        U_6x6 = np.eye(6, dtype=np.complex128)
        U_6x6[0:2, 0:2] = U_term
        Ud = matrix_multiply_6x6(U_6x6, Ud)
        
        # Jy term (trainable, unitless parameter, periodic mod 2π, stored as 2×actual, use parameter/2 in circuit)
        strength_duration_val = strength_durations_array[1, layer]
        a_n = 2.0 * (strength_duration_val / 2.0) * np.array([-sin_kx, -cos_kx, 0.0])
        U_term = pauli_exponentiation_numba(a_n)
        U_6x6 = np.eye(6, dtype=np.complex128)
        U_6x6[0:2, 0:2] = U_term
        Ud = matrix_multiply_6x6(U_6x6, Ud)
        
        # Jz term (trainable, unitless parameter, periodic mod 2π, stored as 2×actual, use parameter/2 in circuit)
        strength_duration_val = strength_durations_array[2, layer]
        a_n = 2.0 * (strength_duration_val / 2.0) * np.array([-sin_ky, -cos_ky, 0.0])
        U_term = pauli_exponentiation_numba(a_n)
        U_6x6 = np.eye(6, dtype=np.complex128)
        U_6x6[0:2, 0:2] = U_term
        Ud = matrix_multiply_6x6(U_6x6, Ud)
        
        # kappa term (trainable, unitless parameter, periodic mod 2π, stored as 2×actual, use parameter/2 in circuit)
        # Multiply by Delta_without_kappa to make it kappa-independent
        strength_duration_val = strength_durations_array[3, layer]
        a_n = np.array([0.0, 0.0, (strength_duration_val / 2.0) * Delta_without_kappa])
        U_term = pauli_exponentiation_numba(a_n)
        U_6x6 = np.eye(6, dtype=np.complex128)
        U_6x6[0:2, 0:2] = U_term
        Ud = matrix_multiply_6x6(U_6x6, Ud)
        
        # g term (trainable, unitless parameter, periodic mod 2π, stored as 2×actual, use parameter/2 in circuit)
        strength_duration_val = strength_durations_array[4, layer]
        a_n = np.array([0.0, -2.0 * (strength_duration_val / 2.0), 0.0])
        U_term = pauli_exponentiation_numba(a_n)
        U_6x6 = np.eye(6, dtype=np.complex128)
        # Apply to submatrices [[0,2],[0,2]] and [[1,3],[1,3]]
        U_6x6[0, 0] = U_term[0, 0]
        U_6x6[0, 2] = U_term[0, 1]
        U_6x6[2, 0] = U_term[1, 0]
        U_6x6[2, 2] = U_term[1, 1]
        
        U_6x6[1, 1] = U_term[0, 0]
        U_6x6[1, 3] = U_term[0, 1]
        U_6x6[3, 1] = U_term[1, 0]
        U_6x6[3, 3] = U_term[1, 1]
        
        Ud = matrix_multiply_6x6(U_6x6, Ud)
        
        # B term (trainable, unitless parameter, periodic mod 2π, stored as 2×actual, use parameter/2 in circuit)
        strength_duration_val = strength_durations_array[5, layer]
        a_n = np.array([0.0, 2.0 * (strength_duration_val / 2.0), 0.0])
        U_term = pauli_exponentiation_numba(a_n)
        U_6x6 = np.eye(6, dtype=np.complex128)
        # Apply to submatrices [[2,4],[2,4]] and [[3,5],[3,5]]
        U_6x6[2, 2] = U_term[0, 0]
        U_6x6[2, 4] = U_term[0, 1]
        U_6x6[4, 2] = U_term[1, 0]
        U_6x6[4, 4] = U_term[1, 1]
        
        U_6x6[3, 3] = U_term[0, 0]
        U_6x6[3, 5] = U_term[0, 1]
        U_6x6[5, 3] = U_term[1, 0]
        U_6x6[5, 5] = U_term[1, 1]
        
        Ud = matrix_multiply_6x6(U_6x6, Ud)
    
    return Ud


def create_variational_circuit(strength_durations, kx, ky):
    """
    Wrapper function that converts dictionary input to array format for Numba
    
    Args:
        strength_durations: Dictionary with keys ['Jx', 'Jy', 'Jz', 'kappa', 'g', 'B'] 
                           and values as arrays of length p (number of layers)
        kx, ky: momentum values
    
    Returns:
        Ud: 6x6 unitary matrix representing the variational circuit
    """
    # Convert dictionary to array format (all 6 parameters are trainable)
    strength_durations_array = np.zeros((6, len(strength_durations['Jx'])), dtype=np.float64)
    strength_durations_array[0, :] = strength_durations['Jx']  # Jx values
    strength_durations_array[1, :] = strength_durations['Jy']  # Jy values
    strength_durations_array[2, :] = strength_durations['Jz']  # Jz values
    strength_durations_array[3, :] = strength_durations['kappa']  # kappa values
    strength_durations_array[4, :] = strength_durations['g']  # g values
    strength_durations_array[5, :] = strength_durations['B']  # B values
    
    # Call Numba-optimized function
    return create_variational_circuit_numba(strength_durations_array, kx, ky, len(strength_durations['Jx']))


def get_k_grid(n_k_points):
    """Create k-grid from -pi to pi (excluding endpoint) for periodic boundary conditions"""
    return np.linspace(-np.pi, np.pi, n_k_points, endpoint=False)


def trotterized_evolution_parameters(num_steps=p):
    """
    Create trotterized evolution parameters based on the original adiabatic evolution
    This provides a starting point for the variational optimization
    All parameters (Jx, Jy, Jz, kappa, g, B) are trainable
    """
    # Time step for trotterization
    dt = T / num_steps
    
    # Initialize strength*duration for all terms
    strength_durations = {
        'Jx': np.ones(num_steps)*dt,
        'Jy': np.ones(num_steps)*dt,
        'Jz': np.ones(num_steps)*dt,
        'kappa': np.ones(num_steps)*dt,
        'g': np.ones(num_steps)*dt,
        'B': np.ones(num_steps)*dt
    }
    
    # Apply time-dependent scaling for g and B terms
    for layer in range(num_steps):
        t = layer * T / num_steps
        g_t = smoothed_g(t)
        B_t = smoothed_B(t)
        strength_durations['g'][layer] *= g_t
        strength_durations['B'][layer] *= B_t
    
    return strength_durations


def randn_strength_durations(std=1.0):
    return {
        'Jx': np.random.randn(p) * std,
        'Jy': np.random.randn(p) * std,
        'Jz': np.random.randn(p) * std,
        'kappa': np.random.randn(p) * std,
        'g': np.random.randn(p) * std,
        'B': np.random.randn(p) * std,
    }


def expand_strength_durations(old_strength_durations, old_p, new_p):
    """
    Expand circuit by inserting zero-valued layers at specified positions
    
    Args:
        old_strength_durations: Dictionary with keys ['Jx', 'Jy', 'Jz', 'kappa', 'g', 'B']
                                and values as arrays of length old_p
        old_p: Current number of layers
        new_p: Target number of layers (must be > old_p)
    
    Returns:
        new_strength_durations: Dictionary with expanded arrays of length new_p
    """
    delta_p = new_p - old_p
    if delta_p <= 0:
        raise ValueError(f"new_p ({new_p}) must be greater than old_p ({old_p})")
    
    # Calculate insertion positions according to the pattern
    if delta_p == 1:
        # Insert at p_old // 2
        insert_positions = [old_p // 2]
    else:
        # For delta_p layers: positions at p_old // (delta_p + 1) * i for i = delta_p, delta_p-1, ..., 1
        # Insert right-to-left (higher indices first)
        insert_positions = []
        divisor = delta_p + 1
        for i in range(delta_p, 0, -1):  # i = delta_p, delta_p-1, ..., 1
            pos = old_p // divisor * i
            insert_positions.append(pos)
    
    # Sort positions in descending order for right-to-left insertion
    insert_positions = sorted(insert_positions, reverse=True)
    
    # Expand each parameter array
    # Start with old array and insert zeros at specified positions (inserting right-to-left)
    new_strength_durations = {}
    for term in ['Jx', 'Jy', 'Jz', 'kappa', 'g', 'B']:
        old_array = old_strength_durations[term].copy()
        current_array = old_array.copy()
        
        # Insert zeros right-to-left (higher positions first) to avoid index shifting issues
        for insert_pos in insert_positions:
            # Insert zero at insert_pos in current_array
            # This shifts everything at and after insert_pos to the right
            current_array = np.insert(current_array, insert_pos, 0.0)
        
        # Verify we got the right size
        if len(current_array) != new_p:
            raise ValueError(f"Size mismatch: expected {new_p}, got {len(current_array)}")
        
        new_strength_durations[term] = current_array
    
    return new_strength_durations


# ===== Simulation Functions =====

def simulate_single_kpoint(kx, ky, strength_durations, n_cycles=1):
    """
    Simulate variational circuit for a single k-point
    
    Args:
        kx, ky: momentum values
        strength_durations: circuit parameters
        n_cycles: number of cooling cycles to perform
    
    Returns:
        final_state: final state after all cycles
        E_diff_list: list of energy differences after each cycle
        E_gs: ground state energy
        single_particle_dm: single-particle density matrix (6x6)
    """
    # Create the variational circuit unitary
    Ud = create_variational_circuit(strength_durations, kx, ky)
    
    # Get the KSL model (we'll use this for the state and ground state energy)
    f = get_f(kx, ky, Jx, Jy, Jz)
    Delta = get_Delta(kx, ky, kappa)
    
    num_cooling_sublattices = 2
    
    # Get the KSL model for state initialization and ground state energy
    hamiltonian, S, E_gs = get_KSL_model(
        f=f, Delta=Delta, g=smoothed_g, B=smoothed_B, 
        initial_state='product', num_cooling_sublattices=num_cooling_sublattices
    )

    # Initialize list to store energy differences after each cycle
    E_diff_list = []
    
    # Perform multiple cooling cycles
    for cycle in range(n_cycles):
        # Reset bath qubits before each cycle (except the first one)
        if cycle > 0:
            S.reset_all_tau()
        
        # Apply the variational circuit
        S.evolve_with_unitary(Ud)
        
        # Calculate energy after this cycle
        E_final = S.get_energy(hamiltonian.get_matrix(T))
        E_diff = E_final - E_gs
        E_diff_list.append(E_diff)
    
    # Get single-particle density matrix
    single_particle_dm = S.matrix
    
    return S, E_diff_list, E_gs, single_particle_dm


def simulate_grid(kx_list, ky_list, strength_durations, n_cycles=1, verbose=False):
    """
    Simulate variational circuit over a momentum grid
    
    Args:
        kx_list, ky_list: momentum space grids
        strength_durations: circuit parameters
        n_cycles: number of cooling cycles to perform
        verbose: whether to print progress
    
    Returns:
        E_diff: 3D array of energy differences (n_cycles, grid_size_x, grid_size_y) or 2D array if n_cycles=1
        single_particle_dm: 4D array of single-particle density matrices
    """
    grid_size_x = len(kx_list)
    grid_size_y = len(ky_list)
    E_diff = np.zeros((n_cycles, grid_size_x, grid_size_y))
    single_particle_dm = np.zeros((grid_size_x, grid_size_y, 6, 6), dtype=complex)
    
    # Loop over momentum space
    for i_kx, kx in enumerate(kx_list):
        if verbose:
            print(f'Processing kx={kx:.3f} ({i_kx+1}/{grid_size_x})')
        for i_ky, ky in enumerate(ky_list):
            # Simulate single k-point
            _, E_diff_list, _, dm = simulate_single_kpoint(kx, ky, strength_durations, n_cycles)
            
            # Store results for all cycles
            for cycle in range(n_cycles):
                E_diff[cycle, i_kx, i_ky] = E_diff_list[cycle]
            single_particle_dm[i_kx, i_ky, :, :] = dm
    
    # If n_cycles=1, return 2D array for backward compatibility
    if n_cycles == 1:
        return E_diff[0, :, :], single_particle_dm
    else:
        return E_diff, single_particle_dm


def simulate_grid_with_analysis(kx_list, ky_list, strength_durations, n_cycles=1):
    """
    Simulate variational circuit over a momentum grid and calculate analysis metrics
    
    Args:
        kx_list, ky_list: momentum space grids
        strength_durations: circuit parameters
        n_cycles: number of cooling cycles to perform
    
    Returns:
        tuple: (E_diff, single_particle_dm, total_chern_number, system_chern_number, bath_chern_number, energy_density)
    """
    # Run simulation
    E_diff, single_particle_dm = simulate_grid(kx_list, ky_list, strength_durations, n_cycles, verbose=False)
    
    # # Calculate Chern numbers
    # total_chern_number = get_chern_number_from_single_particle_dm(single_particle_dm)
    # system_chern_number = get_chern_number_from_single_particle_dm(single_particle_dm[:,:,:2,:2])
    # bath_chern_number = get_chern_number_from_single_particle_dm(single_particle_dm[:,:,2:,2:])
    # # Calculate Chern numbers using exact method
    total_chern_number = chern_from_mixed_spdm(single_particle_dm)
    system_chern_number = chern_from_mixed_spdm(single_particle_dm[:,:,:2,:2])
    bath_chern_number = chern_from_mixed_spdm(single_particle_dm[:,:,2:,2:])
    
    # Calculate average energy density (use final cycle if multiple cycles)
    if n_cycles == 1:
        energy_density = np.nanmean(E_diff) / 2  # E_diff is 2D
    else:
        energy_density = np.nanmean(E_diff[-1, :, :]) / 2  # E_diff is 3D, use final cycle
    
    return E_diff, single_particle_dm, total_chern_number, system_chern_number, bath_chern_number, energy_density


# ===== Parameter Management Functions =====

def strength_durations_to_vector(strength_durations):
    """Convert strength_durations dictionary to a flat vector for optimization"""
    vector = []
    for term in ['Jx', 'Jy', 'Jz', 'kappa', 'g', 'B']:
        vector.extend(strength_durations[term])
    return np.array(vector)


def vector_to_strength_durations(vector):
    """Convert flat vector back to strength_durations dictionary"""
    strength_durations = {}
    start_idx = 0
    for term in ['Jx', 'Jy', 'Jz', 'kappa', 'g', 'B']:
        end_idx = start_idx + p
        strength_durations[term] = vector[start_idx:end_idx]
        start_idx = end_idx
    return strength_durations


# ===== Optimization Functions =====

def objective_function(vector, kx_list, ky_list, n_cycles=1, verbose=False):
    """
    Objective function to minimize: energy density
    
    Args:
        vector: optimization parameters
        kx_list, ky_list: momentum space grids
        n_cycles: number of cooling cycles to perform
        verbose: whether to print progress
    """
    # Convert vector back to strength_durations
    strength_durations = vector_to_strength_durations(vector)
    
    if verbose:
        print(f"    Evaluating objective function on {len(kx_list)}x{len(ky_list)} grid with {n_cycles} cycles...")
    
    # Use helper function to run simulation
    E_diff, _ = simulate_grid(kx_list, ky_list, strength_durations, n_cycles, verbose)
    
    # Calculate average energy density
    # energy_density = np.nanmean(E_diff[-1, :, :]) / 2  # Divide by 2 because we count k and -k together
    # rms energy density
    if n_cycles == 1:
        rms_energy_density = np.sqrt(np.nanmean(E_diff**2)) / 2
    else:
        rms_energy_density = np.sqrt(np.nanmean(E_diff[-1, :, :]**2)) / 2
    
    if verbose:
        # print(f"    Objective function result: {energy_density:.6f}")
        print(f"    RMS energy density: {rms_energy_density:.6f}")
    return rms_energy_density


def optimize_strength_durations(kx_list, ky_list, n_cycles=1, initial_strength_durations=None, method='L-BFGS-B'):
    """
    Optimize strength_durations to minimize energy density (local optimization)
    
    Args:
        kx_list, ky_list: momentum space grid
        n_cycles: number of cooling cycles to perform
        initial_strength_durations: initial guess (if None, uses trotterized evolution)
        method: optimization method ('L-BFGS-B', 'SLSQP', etc.)
    
    Returns:
        optimized_strength_durations: dictionary with optimized parameters
        optimization_result: scipy optimization result
    """
    print("Starting local optimization...")
    
    assert initial_strength_durations is not None
    
    # Convert to vector
    initial_vector = strength_durations_to_vector(initial_strength_durations)
    # initial_vector = np.zeros_like(initial_vector)
    
    bounds = None

    print(f"Optimizing {len(initial_vector)} parameters using {method}")
    print(f"Initial energy density: {objective_function(initial_vector, kx_list, ky_list, n_cycles, verbose=True):.6f}")
    
    # Set up optimization options
    options = {'maxiter': epochs, 'disp': True, 'ftol': 1e-9, 'gtol': 1e-5}
    max_iter = options['maxiter']
    
    # Create a wrapper class to track function evaluations efficiently
    class ObjectiveWrapper:
        def __init__(self, func, kx_list, ky_list, n_cycles):
            self.func = func
            self.kx_list = kx_list
            self.ky_list = ky_list
            self.n_cycles = n_cycles
            self.last_x = None
            self.last_value = None
            self.eval_count = 0
            
        def __call__(self, x):
            self.eval_count += 1
            self.last_x = x.copy()
            self.last_value = self.func(x, self.kx_list, self.ky_list, self.n_cycles, verbose=False)
            return self.last_value
    
    # Create wrapper and callback
    obj_wrapper = ObjectiveWrapper(objective_function, kx_list, ky_list, n_cycles)
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
            # Fallback: compute if not available (shouldn't happen in normal operation)
            current_energy = objective_function(xk, kx_list, ky_list, n_cycles, verbose=False)
            
        print(f"  Iteration {iteration_count[0]}/{max_iter}: Energy density = {current_energy:.6f} (Time: {iteration_time:.2f}s)")
        iteration_start_time[0] = current_time  # Reset for next iteration
        return False
    
    # Perform optimization
    result = minimize(
        obj_wrapper,  # Use the wrapper instead of the raw function
        initial_vector,
        method=method,
        bounds=bounds,
        callback=callback,
        options=options
    )
    
    # Convert result back to strength_durations
    optimized_strength_durations = vector_to_strength_durations(result.x)
    
    print(f"Optimization completed!")
    print(f"Final energy density: {result.fun:.6f}")
    print(f"Success: {result.success}")
    print(f"Iterations: {result.nit}")
    print(f"Function evaluations: {result.nfev}")
    print(f"Gradient evaluations: {result.njev}")
    print(f"Total objective function calls: {obj_wrapper.eval_count}")
    
    return optimized_strength_durations, result


def optimize_strength_durations_global(kx_list, ky_list, n_cycles=1, niter=None, T=1.0, stepsize=0.5):
    """
    Optimize strength_durations to minimize energy density using global optimization (basin hopping)
    
    Args:
        kx_list, ky_list: momentum space grid
        n_cycles: number of cooling cycles to perform
        niter: number of basin hopping iterations (if None, uses epochs)
        T: temperature parameter for basin hopping (default: 1.0)
        stepsize: maximum step size for random displacement (default: 0.5)
    
    Returns:
        optimized_strength_durations: dictionary with optimized parameters
        optimization_result: scipy optimization result
    """
    print("Starting global optimization (basin hopping)...")
    
    global p
    
    # Determine number of parameters
    num_params = 6 * p  # 6 parameter types (Jx, Jy, Jz, kappa, g, B) × p layers
    
    x0 = np.zeros(num_params)
    
    print(f"Optimizing {num_params} parameters using basin hopping (unconstrained)")
    print(f"Number of iterations: {niter}, Temperature: {T}, Step size: {stepsize}")
    
    # Create a wrapper class to track function evaluations efficiently
    class ObjectiveWrapper:
        def __init__(self, func, kx_list, ky_list, n_cycles):
            self.func = func
            self.kx_list = kx_list
            self.ky_list = ky_list
            self.n_cycles = n_cycles
            self.eval_count = 0
            self.best_value = float('inf')
            self.best_x = None
            
        def __call__(self, x):
            self.eval_count += 1
            value = self.func(x, self.kx_list, self.ky_list, self.n_cycles, verbose=False)
            if value < self.best_value:
                self.best_value = value
                self.best_x = x.copy()
            return value
    
    # Create wrapper
    obj_wrapper = ObjectiveWrapper(objective_function, kx_list, ky_list, n_cycles)
    iteration_count = [0]
    iteration_start_time = [time.time()]
    
    def callback(x, f, accept):
        """
        Callback function for basinhopping
        Receives (x, f, accept) where x is current coordinates, f is function value, 
        and accept is whether the step was accepted
        """
        iteration_count[0] += 1
        current_time = time.time()
        iteration_time = current_time - iteration_start_time[0]
        
        # Get current best value from wrapper
        current_energy = obj_wrapper.best_value
        
        status = "accepted" if accept else "rejected"
        print(f"  Iteration {iteration_count[0]}/{niter}: Best energy density = {current_energy:.6f}, "
              f"Current = {f:.6f} ({status}) (Time: {iteration_time:.2f}s, Evals: {obj_wrapper.eval_count})")
        iteration_start_time[0] = current_time  # Reset for next iteration
    
    # Set up local minimizer options (unconstrained - using BFGS instead of L-BFGS-B)
    minimizer_kwargs = {
        "method": "BFGS",
        "options": {"maxiter": 100, "ftol": 1e-6}
    }
    
    # Perform global optimization using basin hopping
    # Note: Reproducibility is ensured by np.random.seed(42) set before generating x0
    result = basinhopping(
        obj_wrapper,
        x0=x0,
        niter=niter,
        T=T,
        stepsize=stepsize,
        minimizer_kwargs=minimizer_kwargs,
        callback=callback
    )
    
    # Convert result back to strength_durations
    optimized_strength_durations = vector_to_strength_durations(result.x)
    
    print(f"Global optimization completed!")
    print(f"Final energy density: {result.fun:.6f}")
    print(f"Function evaluations: {result.nfev}")
    print(f"Total objective function calls: {obj_wrapper.eval_count}")
    print(f"Number of local minimizations: {result.nit}")
    
    return optimized_strength_durations, result


# ===== I/O Functions =====

def save_optimized_parameters_for_res_p(strength_durations, res, p, output_dir=None):
    """
    Save optimized parameters for a specific (res, p) combination
    
    Args:
        strength_durations: Dictionary with optimized parameters
        res: resolution parameter
        p: number of layers
        output_dir: directory to save parameters (default: DATA_DIR/optimized_parameters)
    
    Returns:
        filename: path to saved file
    """
    if output_dir is None:
        output_dir = os.path.join(DATA_DIR, 'optimized_parameters')
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.join(output_dir, f'optimized_params_res{res}_p{p}.npz')
    np.savez(filename, **strength_durations)
    return filename


def load_optimized_parameters_for_res_p(res, p, input_dir=None):
    """
    Load optimized parameters for a specific (res, p) combination
    
    Args:
        res: resolution parameter
        p: number of layers
        input_dir: directory containing saved parameters (default: DATA_DIR/optimized_parameters)
    
    Returns:
        strength_durations: Dictionary with loaded parameters
    """
    if input_dir is None:
        input_dir = os.path.join(DATA_DIR, 'optimized_parameters')
    filename = os.path.join(input_dir, f'optimized_params_res{res}_p{p}.npz')
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Parameters file not found: {filename}")
    
    data = np.load(filename)
    strength_durations = {}
    for term in ['Jx', 'Jy', 'Jz', 'kappa', 'g', 'B']:
        strength_durations[term] = data[term]
    print(f"Loaded optimized parameters from {filename}")
    return strength_durations


def load_optimized_parameters(filename=None):
    """Load previously optimized parameters from file"""
    if filename is None:
        filename = os.path.join(DATA_DIR, 'optimized_strength_durations.npz')
    data = np.load(filename)
    strength_durations = {}
    for term in ['Jx', 'Jy', 'Jz', 'kappa', 'g', 'B']:
        strength_durations[term] = data[term]
    print(f"Loaded optimized parameters from {filename}")
    return strength_durations


# ===== Evaluation Functions =====

def evaluate_loaded_parameters(res_val, p_val, parameter_file=None, input_dir=None, 
                                n_k_points_test_fixed=None, n_cycles_train_eval=None, n_cycles_test_eval=None,
                                plot=True):
    """
    Load circuit parameters from a file and evaluate them on training and test grids
    
    Args:
        res_val: resolution parameter (n_k_points_train = 6*res_val)
        p_val: number of layers in the circuit
        parameter_file: full path to parameter file (if None, uses load_optimized_parameters_for_res_p)
        input_dir: directory containing saved parameters (used if parameter_file is None)
        n_k_points_test_fixed: fixed test grid size (if None, uses global n_k_points_test)
        n_cycles_train_eval: number of cycles for training evaluation (if None, uses n_cycles_train)
        n_cycles_test_eval: number of cycles for test evaluation (if None, uses n_cycles_test)
        plot: whether to plot results for both grids
    
    Returns:
        tuple: (energy_density_train, energy_density_test, strength_durations)
    """
    global p, n_k_points_train, n_k_points_test
    
    # Load parameters
    if parameter_file is not None:
        if not os.path.exists(parameter_file):
            raise FileNotFoundError(f"Parameter file not found: {parameter_file}")
        data = np.load(parameter_file)
        strength_durations = {}
        for term in ['Jx', 'Jy', 'Jz', 'kappa', 'g', 'B']:
            strength_durations[term] = data[term]
        print(f"Loaded parameters from {parameter_file}")
    else:
        if input_dir is None:
            input_dir = os.path.join(DATA_DIR, 'optimized_parameters')
        strength_durations = load_optimized_parameters_for_res_p(res_val, p_val, input_dir)
    
    # Verify p matches
    loaded_p = len(strength_durations['Jx'])
    if loaded_p != p_val:
        print(f"Warning: Parameter file has p={loaded_p}, but requested p={p_val}. Using p={loaded_p}")
        p_val = loaded_p
    
    # Store original values
    old_p = p
    old_n_k_points_train = n_k_points_train
    old_n_k_points_test = n_k_points_test
    
    try:
        # Set new parameters
        p = p_val
        n_k_points_train = 6 * res_val
        if n_k_points_test_fixed is None:
            n_k_points_test_fixed = n_k_points_test
        else:
            n_k_points_test = n_k_points_test_fixed
        
        # Use provided cycle counts or defaults
        if n_cycles_train_eval is None:
            n_cycles_train_eval = n_cycles_train
        if n_cycles_test_eval is None:
            n_cycles_test_eval = n_cycles_test
        
        print(f"\n{'='*60}")
        print(f"EVALUATING LOADED PARAMETERS")
        print(f"{'='*60}")
        print(f"res={res_val}, p={p_val}")
        print(f"Training grid: {n_k_points_train}x{n_k_points_train}")
        print(f"Test grid: {n_k_points_test}x{n_k_points_test}")
        print(f"Training cycles: {n_cycles_train_eval}, Test cycles: {n_cycles_test_eval}")
        print(f"{'='*60}")
        
        # Create training momentum space grid
        kx_list_train = get_k_grid(n_k_points_train)
        ky_list_train = get_k_grid(n_k_points_train)
        
        # Evaluate on training grid
        E_diff_train, single_particle_dm_train, total_chern_number_train, system_chern_number_train, \
        bath_chern_number_train, energy_density_train = simulate_grid_with_analysis(
            kx_list_train, ky_list_train, strength_durations, 
            n_cycles=n_cycles_train_eval
        )
        
        # Print and plot training results
        print_results(energy_density_train, total_chern_number_train, system_chern_number_train,
                     bath_chern_number_train, grid_size=f"{n_k_points_train}x{n_k_points_train} (train, {n_cycles_train_eval} cycles)",
                     phase_name="TRAINING GRID RESULTS")
        
        if plot:
            # Use final cycle for plotting if multiple cycles
            E_diff_train_plot = E_diff_train[-1, :, :] if E_diff_train.ndim == 3 else E_diff_train
            plot_results(E_diff_train_plot, strength_durations, n_k_points_train, "Training ", show_training_points=False)
        
        # Evaluate on test grid
        kx_list_test = get_k_grid(n_k_points_test)
        ky_list_test = get_k_grid(n_k_points_test)
        E_diff_test, single_particle_dm_test, total_chern_number_test, system_chern_number_test, \
        bath_chern_number_test, energy_density_test = simulate_grid_with_analysis(
            kx_list_test, ky_list_test, strength_durations, 
            n_cycles=n_cycles_test_eval
        )
        
        # Print and plot test results
        print_results(energy_density_test, total_chern_number_test, system_chern_number_test,
                     bath_chern_number_test, grid_size=f"{n_k_points_test}x{n_k_points_test} (test, {n_cycles_test_eval} cycles)",
                     phase_name="TEST GRID RESULTS")
        
        if plot:
            # Use final cycle for plotting if multiple cycles
            E_diff_test_plot = E_diff_test[-1, :, :] if E_diff_test.ndim == 3 else E_diff_test
            plot_results(E_diff_test_plot, strength_durations, n_k_points_test, "Test ", show_training_points=True)
            
            # Plot energy density vs cycles for both grids
            max_cycles = max(n_cycles_train_eval, n_cycles_test_eval)
            plot_energy_density_vs_cycles(
                kx_list_train, ky_list_train, kx_list_test, ky_list_test, 
                strength_durations, max_cycles=max_cycles
            )
        
        return energy_density_train, energy_density_test, strength_durations
        
    finally:
        # Restore original values
        p = old_p
        n_k_points_train = old_n_k_points_train
        n_k_points_test = old_n_k_points_test


# ===== Plotting Functions =====

def plot_results(E_diff, optimized_strength_durations, grid_size, title_suffix="", show_training_points=False):
    """
    Unified plotting function for results visualization
    
    Args:
        E_diff: Energy difference array
        optimized_strength_durations: Dictionary of optimized parameters
        grid_size: Size of the momentum grid used
        title_suffix: Additional text for plot titles
        show_training_points: Whether to highlight training data points (for test grid)
    """
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Energy difference heatmap
    plt.subplot(1, 2, 1)
    
    # Create coordinate arrays for pcolormesh
    kx_coords = np.linspace(-np.pi, np.pi, grid_size + 1)[:-1]
    ky_coords = np.linspace(-np.pi, np.pi, grid_size + 1)[:-1]
    
    # Use pcolormesh instead of imshow for better coordinate control
    # Set color scale to start at 0
    vmin = 0.0
    vmax = np.max(E_diff)
    plt.pcolormesh(kx_coords, ky_coords, E_diff, vmin=vmin, vmax=vmax)
    plt.colorbar(label='Energy difference')
    
    # Add training points overlay if requested and this is a test grid
    if show_training_points and grid_size == n_k_points_test:
        # Calculate exact indices for training points
        # Training grid: 7x7, Test grid: 19x19
        # The training points should be evenly spaced in the test grid
        
        # Calculate the step size to get exactly 7 points from 19
        step = (grid_size - 1) // (n_k_points_train - 1)  # (19-1)/(7-1) = 18/6 = 3
        
        # Get the exact indices for training points
        training_indices = np.arange(0, grid_size, step)
        
        # Ensure we have exactly 7 points
        if len(training_indices) > n_k_points_train:
            training_indices = training_indices[:n_k_points_train]
        elif len(training_indices) < n_k_points_train:
            # If we don't have enough points, add the last point
            training_indices = np.append(training_indices, grid_size - 1)
        
        # Get the exact coordinates for training points
        kx_train_coords = kx_coords[training_indices]
        ky_train_coords = ky_coords[training_indices]
        
        # Plot training points as red crosses - these will be perfectly aligned
        for i, kx_coord in enumerate(kx_train_coords):
            for j, ky_coord in enumerate(ky_train_coords):
                plt.scatter(kx_coord, ky_coord, c='red', marker='x', s=50, linewidths=2, 
                           label='Training points' if i==0 and j==0 else "")
    
    plt.xlabel('kx')
    plt.ylabel('ky')
    plt.title(f'Energy difference {title_suffix}(Grid: {grid_size}x{grid_size})')
    if show_training_points and grid_size == n_k_points_test:
        plt.legend()
    
    # Plot 2: Optimized parameters
    plt.subplot(1, 2, 2)
    terms = ['Jx', 'Jy', 'Jz', 'kappa', 'g', 'B']
    for i, term in enumerate(terms):
        plt.plot(optimized_strength_durations[term], label=term, marker='o')
    plt.xlabel('Layer')
    plt.ylabel('Strength*duration')
    plt.title('Optimized strength*duration parameters')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


def print_results(energy_density, total_chern_number, system_chern_number, 
                 bath_chern_number, opt_result=None, grid_size=None, phase_name="RESULTS"):
    """
    Unified function to print results
    
    Args:
        energy_density: Average energy density
        total_chern_number: Total Chern number
        system_chern_number: System Chern number
        bath_chern_number: Bath Chern number
        opt_result: Optimization result object (optional)
        grid_size: Size of momentum grid (optional)
        phase_name: Name of the phase (e.g., "TRAINING", "TESTING", "FINAL RESULTS")
    """
    print(f"\n" + "="*60)
    print(phase_name)
    print("="*60)
    
    if grid_size:
        if isinstance(grid_size, str):
            print(f"Grid size: {grid_size}")
        else:
            print(f"Grid size: {grid_size}x{grid_size} = {grid_size**2} points")
    
    print(f"Energy density: {energy_density:.6f}")
    print(f"Total Chern number: {total_chern_number:.6f}")
    print(f"System Chern number: {system_chern_number:.6f}")
    print(f"Bath Chern number: {bath_chern_number:.6f}")
    
    if opt_result:
        print(f"Optimization success: {opt_result.success}")
        print(f"Optimization iterations: {opt_result.nit}")
    
    print("="*60)


def evaluate_initialization(strength_durations, kx_list_train, ky_list_train, n_cycles_eval=1, check_monotonic=True):
    """
    Evaluate a single initialization by computing its steady state energy density
    
    Args:
        strength_durations: circuit parameters to evaluate
        kx_list_train, ky_list_train: training momentum grids
        n_cycles_eval: number of cycles to use for evaluation
        check_monotonic: whether to check for monotonic decrease between cycles 2-5
    
    Returns:
        energy_density: average energy density for this initialization
        is_monotonic: whether energy decreases monotonically between cycles 2-5
    """
    # Use helper function to run simulation
    E_diff, _ = simulate_grid(kx_list_train, ky_list_train, strength_durations, n_cycles_eval, verbose=False)
    
    # Calculate average energy density
    if n_cycles_eval == 1:
        energy_density = np.nanmean(E_diff) / 2
        is_monotonic = True  # Single cycle, no monotonic check needed
    else:
        energy_density = np.nanmean(E_diff[-1, :, :]) / 2
        
        # Check monotonic decrease between cycles 2-5
        is_monotonic = True
        if check_monotonic and n_cycles_eval >= 5:
            # Calculate energy density for each cycle
            cycle_energies = []
            for cycle in range(n_cycles_eval):
                cycle_energy = np.nanmean(E_diff[cycle, :, :]) / 2
                cycle_energies.append(cycle_energy)
            
            # Check if energy decreases monotonically between cycles 2-5 (indices 1-4)
            for i in range(1, 4):  # Check cycles 2,3,4 vs previous
                if cycle_energies[i] >= cycle_energies[i-1]:
                    is_monotonic = False
                    break
    
    return energy_density, is_monotonic


def find_best_initialization(kx_list_train, ky_list_train, n_random_trials=10, n_cycles_eval=1, std=1.0, require_monotonic=True):
    """
    Test multiple random initializations and select the one with lowest energy density
    Optionally filter for monotonic energy decrease between cycles 2-5
    
    Args:
        kx_list_train, ky_list_train: training momentum grids
        n_random_trials: number of random initializations to test
        n_cycles_eval: number of cycles to use for evaluation
        std: standard deviation for random initialization
        require_monotonic: whether to only accept monotonic initializations
    
    Returns:
        best_strength_durations: initialization with lowest energy density
        best_energy_density: energy density of the best initialization
        all_energies: list of all energy densities tested
        monotonic_count: number of monotonic initializations found
    """
    print(f"\nTesting {n_random_trials} random initializations...")
    print(f"Using {n_cycles_eval} cycle(s) for evaluation")
    if require_monotonic and n_cycles_eval >= 5:
        print("Requiring monotonic energy decrease between cycles 2-5")
    
    best_strength_durations = None
    best_energy_density = float('inf')
    all_energies = []
    monotonic_count = 0
    rejected_count = 0
    
    for trial in range(n_random_trials):
        # Generate random initialization
        strength_durations = randn_strength_durations(std=std)
        
        # Evaluate this initialization
        energy_density, is_monotonic = evaluate_initialization(strength_durations, kx_list_train, ky_list_train, n_cycles_eval, check_monotonic=require_monotonic)
        
        # Check if this initialization meets our criteria
        if require_monotonic and n_cycles_eval >= 5 and not is_monotonic:
            rejected_count += 1
            print(f"  Trial {trial+1}/{n_random_trials}: Energy density = {energy_density:.6f} (REJECTED - not monotonic)")
            continue
        
        all_energies.append(energy_density)
        if is_monotonic:
            monotonic_count += 1
        
        print(f"  Trial {trial+1}/{n_random_trials}: Energy density = {energy_density:.6f} {'(monotonic)' if is_monotonic else ''}")
        
        # Keep track of the best one
        if energy_density < best_energy_density:
            best_energy_density = energy_density
            best_strength_durations = strength_durations.copy()
    
    print(f"\nInitialization search completed:")
    print(f"  Total trials: {n_random_trials}")
    print(f"  Accepted: {len(all_energies)}")
    print(f"  Rejected (non-monotonic): {rejected_count}")
    print(f"  Monotonic: {monotonic_count}")
    print(f"  Best energy density: {best_energy_density:.6f}")
    if len(all_energies) > 1:
        print(f"  Improvement over worst: {max(all_energies) - best_energy_density:.6f}")
        print(f"  Standard deviation of accepted: {np.std(all_energies):.6f}")
    
    return best_strength_durations, best_energy_density, all_energies, monotonic_count


# ===== Main Workflow Functions =====

def train_variational_circuit():
    """
    Train the variational circuit using a smaller momentum grid
    """
    print("="*60)
    print("TRAINING PHASE")
    print("="*60)
    
    # Create training momentum space grid
    kx_list_train = get_k_grid(n_k_points_train)
    ky_list_train = get_k_grid(n_k_points_train)
    
    print(f"Training grid: {len(kx_list_train)}x{len(ky_list_train)} = {len(kx_list_train)*len(ky_list_train)} points")
    
    # Get trotterized evolution parameters as initialization
    initial_strength_durations = trotterized_evolution_parameters()
    
    print(f"Initial strength*duration parameters shape: {[(k, v.shape) for k, v in initial_strength_durations.items()]}")
    
    # Optimize the strength_durations on training data
    optimized_strength_durations, opt_result = optimize_strength_durations(
        kx_list_train, ky_list_train[ky_list_train >= 0], 
        n_cycles=n_cycles_train,
        initial_strength_durations=initial_strength_durations,
        method='L-BFGS-B'
    )
    
    # Save optimized parameters
    save_path = os.path.join(DATA_DIR, 'optimized_strength_durations.npz')
    np.savez(save_path, **optimized_strength_durations)
    print(f"Optimized parameters saved to '{save_path}'")
    
    return optimized_strength_durations, opt_result


def plot_energy_density_vs_cycles(kx_list_train, ky_list_train, kx_list_test, ky_list_test, strength_durations, max_cycles=50):
    """
    Plot energy density as a function of the number of cooling cycles for both training and test grids
    
    Args:
        kx_list_train, ky_list_train: training momentum space grids
        kx_list_test, ky_list_test: test momentum space grids
        strength_durations: circuit parameters
        max_cycles: maximum number of cycles to test
    """
    print(f"\nAnalyzing energy density vs cycles (up to {max_cycles} cycles)...")
    
    # Run simulation for training grid
    print(f"  Running simulation on training grid ({len(kx_list_train)}x{len(ky_list_train)}) with {max_cycles} cycles...")
    E_diff_train_all_cycles, _ = simulate_grid(kx_list_train, ky_list_train, strength_durations, max_cycles, verbose=False)
    
    # Run simulation for test grid
    print(f"  Running simulation on test grid ({len(kx_list_test)}x{len(ky_list_test)}) with {max_cycles} cycles...")
    E_diff_test_all_cycles, _ = simulate_grid(kx_list_test, ky_list_test, strength_durations, max_cycles, verbose=False)
    
    # Calculate energy densities for each cycle count
    cycle_counts = range(1, max_cycles + 1)
    energy_densities_train = []
    energy_densities_test = []
    
    for n_cycles in cycle_counts:
        # Training grid
        E_diff_train = E_diff_train_all_cycles[n_cycles - 1, :, :]
        energy_density_train = np.nanmean(E_diff_train) / 2
        energy_densities_train.append(energy_density_train)
        
        # Test grid
        E_diff_test = E_diff_test_all_cycles[n_cycles - 1, :, :]
        energy_density_test = np.nanmean(E_diff_test) / 2
        energy_densities_test.append(energy_density_test)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.plot(cycle_counts, energy_densities_train, 'b-o', linewidth=2, markersize=6, label=f'Training Grid ({len(kx_list_train)}x{len(ky_list_train)})')
    plt.plot(cycle_counts, energy_densities_test, 'r-s', linewidth=2, markersize=6, label=f'Test Grid ({len(kx_list_test)}x{len(ky_list_test)})')
    
    plt.xlabel('Number of Cooling Cycles')
    plt.ylabel('Energy Density')
    plt.title('Energy Density vs Number of Cooling Cycles (Training vs Test Data)')
    plt.grid(True, alpha=0.3)
    
    # Add horizontal lines at final values for reference
    plt.axhline(y=energy_densities_train[-1], color='b', linestyle='--', alpha=0.7, 
                label=f'Training final: {energy_densities_train[-1]:.6f}')
    plt.axhline(y=energy_densities_test[-1], color='r', linestyle='--', alpha=0.7, 
                label=f'Test final: {energy_densities_test[-1]:.6f}')
    
    # Add vertical line at the training cycles for reference
    plt.axvline(x=n_cycles_train, color='g', linestyle=':', alpha=0.7, 
                label=f'Training cycles: {n_cycles_train}')
    
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Print statistics for both grids
    print(f"\nEnergy density statistics:")
    print(f"Training Grid ({len(kx_list_train)}x{len(ky_list_train)}):")
    print(f"  After 1 cycle: {energy_densities_train[0]:.6f}")
    print(f"  After {n_cycles_train} cycles (training): {energy_densities_train[n_cycles_train-1]:.6f}")
    print(f"  After {max_cycles} cycles (final): {energy_densities_train[-1]:.6f}")
    print(f"  Improvement from 1 to {max_cycles} cycles: {energy_densities_train[0] - energy_densities_train[-1]:.6f}")
    
    print(f"\nTest Grid ({len(kx_list_test)}x{len(ky_list_test)}):")
    print(f"  After 1 cycle: {energy_densities_test[0]:.6f}")
    print(f"  After {n_cycles_train} cycles (training): {energy_densities_test[n_cycles_train-1]:.6f}")
    print(f"  After {max_cycles} cycles (final): {energy_densities_test[-1]:.6f}")
    print(f"  Improvement from 1 to {max_cycles} cycles: {energy_densities_test[0] - energy_densities_test[-1]:.6f}")
    
    return cycle_counts, energy_densities_train, energy_densities_test


# ===== Progressive Circuit Expansion Functions =====

def run_single_experiment(p_val, kx_list_train, ky_list_train, kx_list_test, ky_list_test, 
                         initial_strength_durations=None, use_global=False):
    """
    Run a single experiment with given parameters
    
    Args:
        p_val: number of layers in the circuit
        kx_list_train, ky_list_train: training momentum space grids
        kx_list_test, ky_list_test: test momentum space grids
        initial_strength_durations: optional initial parameters (if None, uses trotterized)
        use_global: if True, use global optimization (basin hopping); if False, use local optimization
    
    Returns:
        tuple: (energy_density_train, energy_density_test, optimized_strength_durations)
    """
    global p
    
    # Store original value
    old_p = p
    
    try:
        # Set new parameter
        p = p_val
        
        if use_global:
            # Use global optimization (doesn't require initial guess)
            print(f"Using global optimization for p={p_val}")
            optimized_strength_durations, opt_result = optimize_strength_durations_global(
                kx_list_train, ky_list_train[ky_list_train >= 0], 
                n_cycles=n_cycles_train,
                niter=50,
            )
        else:
            # Use local optimization (requires initial guess)
            # Get initialization parameters
            if initial_strength_durations is None:
                initial_strength_durations = trotterized_evolution_parameters(num_steps=p)
            
            # Optimize the strength_durations on training data
            optimized_strength_durations, opt_result = optimize_strength_durations(
                kx_list_train, ky_list_train[ky_list_train >= 0], 
                n_cycles=n_cycles_train,
                initial_strength_durations=initial_strength_durations,
                method='L-BFGS-B'
            )
        
        # Evaluate on training grid (using n_cycles_test)
        _, _, _, _, _, energy_density_train = simulate_grid_with_analysis(
            kx_list_train, ky_list_train, optimized_strength_durations, 
            n_cycles=n_cycles_test
        )
        
        # Evaluate on test grid (using n_cycles_test)
        _, _, _, _, _, energy_density_test = simulate_grid_with_analysis(
            kx_list_test, ky_list_test, optimized_strength_durations, 
            n_cycles=n_cycles_test
        )
        
        return energy_density_train, energy_density_test, optimized_strength_durations
        
    finally:
        # Restore original value
        p = old_p


def run_progressive_circuit_expansion(output_csv=None, params_output_dir=None,
                                     trotterized_csv=None):
    """
    Run progressive circuit expansion over res and p values
    
    Args:
        output_csv: filename for saving optimized results
        params_output_dir: directory to save optimized parameters for each (res, p) combination
        trotterized_csv: filename for saving trotterized initialization results
    
    Returns:
        tuple: (optimized_results, trotterized_results) as lists of dictionaries
    """
    if output_csv is None:
        output_csv = os.path.join(DATA_DIR, 'progressive_circuit_expansion_results.csv')
    if params_output_dir is None:
        params_output_dir = os.path.join(DATA_DIR, 'optimized_parameters')
    if trotterized_csv is None:
        trotterized_csv = os.path.join(DATA_DIR, 'progressive_circuit_expansion_trotterized.csv')
    
    # Parameter ranges
    res_values = [1, 2, 3, 4]
    p_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Get the smallest p value (will use global optimization for this)
    smallest_p = min(p_values)
    
    # Fixed test grid size (use current default)
    n_k_points_test_fixed = 6*20  # 120
    
    # Store results
    results = []
    trotterized_results = []
    
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
    previous_optimized = {}  # key: res, value: (prev_p, optimized_strength_durations)
    
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
                    energy_density_train, energy_density_test, optimized_strength_durations = run_single_experiment(
                        p_val, kx_list_train, ky_list_train, kx_list_test, ky_list_test, 
                        use_global=True
                    )
                elif res in previous_optimized:
                    prev_p, prev_optimized = previous_optimized[res]
                    if p_val > prev_p:
                        # Expand previous optimized circuit by inserting zero layers
                        print(f"  Expanding circuit from p={prev_p} to p={p_val} and using LOCAL optimization...")
                        initial_params = expand_strength_durations(prev_optimized, prev_p, p_val)
                        energy_density_train, energy_density_test, optimized_strength_durations = run_single_experiment(
                            p_val, kx_list_train, ky_list_train, kx_list_test, ky_list_test, 
                            initial_strength_durations=initial_params, use_global=False
                        )
                    else:
                        # p decreased (shouldn't happen), restart with global optimization
                        raise ValueError(f"p decreased from {prev_p} to {p_val}, this shouldn't happen")
                else:
                    # This shouldn't happen if we handle smallest_p correctly above
                    raise ValueError(f"No previous optimization found for res={res}, p={p_val}, and p != smallest_p")
                
                # Store optimized parameters for next iteration
                previous_optimized[res] = (p_val, optimized_strength_durations)
                
                # Save optimized parameters to file
                param_filename = save_optimized_parameters_for_res_p(
                    optimized_strength_durations, res, p_val, params_output_dir
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
    print(f"Trotterized results saved to: {trotterized_csv}")
    print(f"{'='*60}\n")
    
    return results, trotterized_results


def save_results_to_csv(results, filename):
    """
    Save results to CSV file
    
    Args:
        results: list of dictionaries with results
        filename: output CSV filename
    """
    if not results:
        return
    
    fieldnames = ['res', 'n_k_points_train', 'p', 'energy_density_train', 'energy_density_test']
    
    # Check if error field exists in any result
    if any('error' in r for r in results):
        fieldnames.append('error')
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

def main():
    """Main function to run the variational circuit simulation with training and testing"""
    print("Starting variational circuit simulation with training and testing...")
    print(f"Using {p} layers for variational circuit")
    print(f"Training with {n_cycles_train} cooling cycles")
    print(f"Testing with {n_cycles_test} cooling cycles")
    
    # Phase 1: Training
    optimized_strength_durations, opt_result = train_variational_circuit()
    
    # Phase 2: Testing
    print("\n" + "="*60)
    print("TESTING PHASE")
    print("="*60)
    kx_list_test = get_k_grid(n_k_points_test)
    ky_list_test = get_k_grid(n_k_points_test)
    E_diff_test, single_particle_dm_test, total_chern_number_test, system_chern_number_test, bath_chern_number_test, energy_density_test = simulate_grid_with_analysis(
        kx_list_test, ky_list_test, optimized_strength_durations, n_cycles_test
    )

    # Phase 2.5: Training grid simulation
    print("\n" + "="*60)
    print("TRAINING GRID SIMULATION")
    print("="*60)
    kx_list_train = get_k_grid(n_k_points_train)
    ky_list_train = get_k_grid(n_k_points_train)
    E_diff_train, single_particle_dm_train, total_chern_number_train, system_chern_number_train, bath_chern_number_train, energy_density_train = simulate_grid_with_analysis(
        kx_list_train, ky_list_train, optimized_strength_durations, n_cycles_test
    )

    # Phase 3: Energy density vs cycles analysis
    print("\n" + "="*60)
    print("ENERGY DENSITY VS CYCLES ANALYSIS")
    print("="*60)
    
    # Create momentum space grids for the analysis
    kx_list_train = get_k_grid(n_k_points_train)
    ky_list_train = get_k_grid(n_k_points_train)
    kx_list_test = get_k_grid(n_k_points_test)
    ky_list_test = get_k_grid(n_k_points_test)
    
    # Plot energy density vs cycles for both grids
    cycle_counts, energy_densities_train, energy_densities_test = plot_energy_density_vs_cycles(
        kx_list_train, ky_list_train, kx_list_test, ky_list_test, optimized_strength_durations, max_cycles=n_cycles_test
    )

    # Print results for both grids
    print_results(energy_density_train, total_chern_number_train, system_chern_number_train, 
                bath_chern_number_train, opt_result=opt_result, 
                grid_size=f"{n_k_points_train}x{n_k_points_train} (train, {n_cycles_test} cycles)", 
                phase_name="TRAINING GRID RESULTS")
    
    print_results(energy_density_test, total_chern_number_test, system_chern_number_test, 
                bath_chern_number_test, opt_result=opt_result, 
                grid_size=f"{n_k_points_test}x{n_k_points_test} (test, {n_cycles_test} cycles)", 
                phase_name="TEST GRID RESULTS")
    
    # Plot results for both grids
    E_diff_train_plot = E_diff_train[-1, :, :] if E_diff_train.ndim == 3 else E_diff_train
    plot_results(E_diff_train_plot, optimized_strength_durations, n_k_points_train, "Training ", show_training_points=False)
    
    E_diff_test_plot = E_diff_test[-1, :, :] if E_diff_test.ndim == 3 else E_diff_test
    plot_results(E_diff_test_plot, optimized_strength_durations, n_k_points_test, "Test ", show_training_points=True)
    

if __name__ == "__main__":
    import sys
    
    # Check if user wants to run progressive circuit expansion
    if True:
        output_file = sys.argv[2] if len(sys.argv) > 2 else os.path.join(DATA_DIR, 'progressive_circuit_expansion_results.csv')
        trot_file = sys.argv[3] if len(sys.argv) > 3 else os.path.join(DATA_DIR, 'progressive_circuit_expansion_trotterized.csv')
        run_progressive_circuit_expansion(output_csv=output_file, trotterized_csv=trot_file)
    else:
        main()
