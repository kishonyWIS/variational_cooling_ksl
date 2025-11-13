from itertools import product
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import time
import sys
import os

# Add project root to path to access root-level dependencies
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from time_dependence_functions import get_g, get_B
from translational_invariant_KSL import get_KSL_model, get_Delta, get_f

# Import get_k_grid from variational_circuit_KSL_numba
sys.path.insert(0, os.path.dirname(__file__))
from variational_circuit_KSL_numba import get_k_grid

# Data directory path (relative to this file)
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')

# Set random seed for reproducibility
# np.random.seed(42)

# Pauli matrices
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
p = 10  # Number of layers in the variational circuit
n_k_points_train = 1 + 6*1  # Training grid size
n_k_points_test = 1 + 6*1   # Testing grid size

# Cooling cycle parameters
n_cycles_train = 5    # Number of cooling cycles for training
n_cycles_test = 10     # Number of cooling cycles for testing
epochs = 1000


def get_chern_number_from_single_particle_dm(single_particle_dm):
    """Calculate Chern number from single-particle density matrix"""
    dP_dkx = np.diff(single_particle_dm, axis=0)[:,:-1,:,:]
    dP_dky = np.diff(single_particle_dm, axis=1)[:-1,:,:,:]
    P = single_particle_dm[:-1,:-1,:,:]
    integrand = np.zeros(P.shape[0:2],dtype=complex)
    for i_kx, i_ky in product(range(P.shape[0]), repeat=2):
        integrand[i_kx,i_ky] = np.trace(P[i_kx,i_ky,:,:] @ (dP_dkx[i_kx,i_ky,:,:] @ dP_dky[i_kx,i_ky,:,:] - dP_dky[i_kx,i_ky,:,:] @ dP_dkx[i_kx,i_ky,:,:]))
    return (np.sum(integrand)/(2*np.pi)).imag

def pauli_exponentiation(a_n):
    """
    Compute exp(i * a * (n_vec · σ)) using the formula:
    exp(i * a * (n_vec · σ)) = I * cos(a) + i * (n_vec · σ) * sin(a)
    
    Args:
        a_n: unormalized vector a*n_vec
    
    Returns:
        2x2 unitary matrix
    """

    a = np.linalg.norm(a_n)
    if a < 1e-10:
        return np.eye(2, dtype=complex)
    n_vec = a_n / a
    
    # Identity matrix
    I = np.eye(2, dtype=complex)
    
    # Construct n_vec · σ
    n_dot_sigma = n_vec[0] * sigma_x + n_vec[1] * sigma_y + n_vec[2] * sigma_z
    
    # Apply the formula
    return I * np.cos(a) + 1j * n_dot_sigma * np.sin(a)


def create_variational_circuit(strength_durations, kx, ky):
    """
    Create variational circuit unitary from individual Hamiltonian terms
    
    Args:
        strength_durations: Dictionary with keys ['Jx', 'Jy', 'Jz', 'kappa', 'g', 'B'] 
                           and values as arrays of length p (number of layers)
                           Each value represents the combined strength*duration for that term
        kx, ky: momentum values
    
    Returns:
        Ud: 6x6 unitary matrix representing the variational circuit
    """
    Ud = np.eye(6, dtype=complex)
    
    # Apply p layers
    for layer in range(p):
        # Get the base coefficients (only Delta is still time-independent)
        Delta = get_Delta(kx, ky, kappa)

        # Apply each term with its combined strength*duration for this layer
        for term_name, strength_duration in strength_durations.items():
            strength_duration_val = strength_duration[layer]

            U_6x6 = np.eye(6, dtype=complex)
            
            if term_name in ['Jx', 'Jy', 'Jz', 'kappa']:
                # These terms act on the first 2x2 block [:2, :2]
                if term_name == 'Jx':
                    a_n = np.array([0, -2*Jx * strength_duration_val, 0])
                elif term_name == 'Jy':
                    a_n = 2*Jy * strength_duration_val * np.array([-np.sin(kx), -np.cos(kx), 0])
                elif term_name == 'Jz':
                    a_n = 2*Jz * strength_duration_val * np.array([-np.sin(ky), -np.cos(ky), 0])
                elif term_name == 'kappa':
                    a_n = np.array([0, 0, Delta * strength_duration_val])
                
                U_term = pauli_exponentiation(a_n)
                
                # Start with identity and place the 2x2 unitary in the first block
                U_6x6[:2, :2] = U_term
                
            elif term_name == 'g':
                # g_t scaling is already applied in strength_duration_val
                a_n = np.array([0, -2 * strength_duration_val, 0])
                U_term = pauli_exponentiation(a_n)
                # insert this on the submatrix [[0,2],[0,2]] and on [[1,3],[1,3]]
                U_6x6[np.ix_([0,2],[0,2])] = U_term
                U_6x6[np.ix_([1,3],[1,3])] = U_term

            elif term_name == 'B':
                # B_t scaling is already applied in strength_duration_val
                a_n = np.array([0, 2 * strength_duration_val, 0])
                U_term = pauli_exponentiation(a_n)
                # insert this on the submatrix [[2,4],[2,4]] and on [[3,5],[3,5]]
                U_6x6[np.ix_([2,4],[2,4])] = U_term
                U_6x6[np.ix_([3,5],[3,5])] = U_term
            
            Ud = U_6x6 @ Ud
    
    return Ud

def trotterized_evolution_parameters():
    """
    Create trotterized evolution parameters based on the original adiabatic evolution
    This provides a starting point for the variational optimization
    The time-dependent scaling (g_t and B_t) is applied here, not in circuit construction
    """
    # Time step for trotterization
    dt = T / p
    
    # Initialize strength*duration for each term and layer
    strength_durations = {
        'Jx': np.ones(p)*dt,
        'Jy': np.ones(p)*dt,
        'Jz': np.ones(p)*dt,
        'kappa': np.ones(p)*dt,
        'g': np.ones(p)*dt,
        'B': np.ones(p)*dt
    }
    
    # Apply time-dependent scaling for g and B terms
    for layer in range(p):
        t = layer * T / p
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

def multiple_cooling_cycles_variational(kx, ky, strength_durations, n_cycles=1):
    """
    Perform multiple cooling cycles using the variational circuit
    
    Args:
        kx, ky: momentum values
        strength_durations: circuit parameters
        n_cycles: number of cooling cycles to perform
    
    Returns:
        S: final state after all cycles
        E_diff: list of energy differences after each cycle
        E_gs: ground state energy
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
    
    return S, E_diff_list, E_gs

def run_simulation_over_momentum_grid(kx_list, ky_list, strength_durations, n_cycles=1, verbose=False):
    """
    Helper function to run simulation over a momentum grid and return energy differences
    
    Args:
        kx_list, ky_list: momentum space grids
        strength_durations: circuit parameters
        n_cycles: number of cooling cycles to perform
        verbose: whether to print progress
    
    Returns:
        E_diff: 3D array of energy differences (n_cycles, grid_size_x, grid_size_y) or 2D array if n_cycles=1
        single_particle_dm: 4D array of single-particle density matrices (optional)
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
            # Perform multiple cooling cycles with given parameters
            S, E_diff_list, E_gs = multiple_cooling_cycles_variational(kx, ky, strength_durations, n_cycles)
            
            # Store results for all cycles
            for cycle in range(n_cycles):
                E_diff[cycle, i_kx, i_ky] = E_diff_list[cycle]
            single_particle_dm[i_kx, i_ky, :, :] = S.matrix
    
    # If n_cycles=1, return 2D array for backward compatibility
    if n_cycles == 1:
        return E_diff[0, :, :], single_particle_dm
    else:
        return E_diff, single_particle_dm

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
    E_diff, _ = run_simulation_over_momentum_grid(kx_list, ky_list, strength_durations, n_cycles, verbose)
    
    # Calculate average energy density
    # energy_density = np.nanmean(E_diff[-1, :, :]) / 2  # Divide by 2 because we count k and -k together
    # rms energy density
    rms_energy_density = np.sqrt(np.nanmean(E_diff[-1, :, :]**2)) / 2
    
    if verbose:
        # print(f"    Objective function result: {energy_density:.6f}")
        print(f"    RMS energy density: {rms_energy_density:.6f}")
    return rms_energy_density

def optimize_strength_durations(kx_list, ky_list, n_cycles=1, initial_strength_durations=None, method='L-BFGS-B'):
    """
    Optimize strength_durations to minimize energy density
    
    Args:
        kx_list, ky_list: momentum space grid
        n_cycles: number of cooling cycles to perform
        initial_strength_durations: initial guess (if None, uses trotterized evolution)
        method: optimization method ('L-BFGS-B', 'SLSQP', etc.)
    
    Returns:
        optimized_strength_durations: dictionary with optimized parameters
        optimization_result: scipy optimization result
    """
    print("Starting optimization...")
    
    assert initial_strength_durations is not None
    
    # Convert to vector
    initial_vector = strength_durations_to_vector(initial_strength_durations)
    # initial_vector = np.zeros_like(initial_vector)
    
    # Set bounds: allow both positive and negative values for flexibility
    # but keep them reasonable (e.g., -10 to 10)
    bounds = None#[(-10.0, 10.0)] * len(initial_vector)
    
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
    kx_coords = np.linspace(-np.pi, np.pi, grid_size)
    ky_coords = np.linspace(-np.pi, np.pi, grid_size)
    
    # Use pcolormesh instead of imshow for better coordinate control
    plt.pcolormesh(kx_coords, ky_coords, E_diff)
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

def run_simulation_on_grid(kx_list, ky_list, strength_durations, n_cycles=1, grid_name="", plot=True):
    """
    Run simulation on a given momentum grid
    
    Args:
        kx_list, ky_list: momentum space grids
        strength_durations: circuit parameters
        n_cycles: number of cooling cycles to perform
        grid_name: name for display purposes
        plot: whether to plot results
    
    Returns:
        tuple: (E_diff, single_particle_dm, total_chern_number, system_chern_number, bath_chern_number, energy_density)
    """
    grid_size = len(kx_list)
    print(f"{grid_name} grid: {grid_size}x{grid_size} = {grid_size**2} points")
    
    # Use helper function to run simulation
    E_diff, single_particle_dm = run_simulation_over_momentum_grid(kx_list, ky_list, strength_durations, n_cycles, verbose=False)
    
    # Calculate Chern numbers
    total_chern_number = get_chern_number_from_single_particle_dm(single_particle_dm)
    system_chern_number = get_chern_number_from_single_particle_dm(single_particle_dm[:,:,:2,:2])
    bath_chern_number = get_chern_number_from_single_particle_dm(single_particle_dm[:,:,2:,2:])
    
    # Calculate average energy density (use final cycle if multiple cycles)
    if n_cycles == 1:
        energy_density = np.nanmean(E_diff) / 2  # E_diff is 2D
    else:
        energy_density = np.nanmean(E_diff[-1, :, :]) / 2  # E_diff is 3D, use final cycle
    
    if plot:
        # Use final cycle for plotting
        E_diff_plot = E_diff[-1, :, :] if n_cycles > 1 else E_diff
        plot_results(E_diff_plot, strength_durations, grid_size, f"{grid_name} ")
    
    return E_diff, single_particle_dm, total_chern_number, system_chern_number, bath_chern_number, energy_density

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
    
    # Get initial trotterized evolution parameters
    initial_strength_durations = randn_strength_durations()
    # initial_strength_durations = trotterized_evolution_parameters()
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

def test_variational_circuit(optimized_strength_durations):
    """
    Test the variational circuit using a larger momentum grid
    """
    print("\n" + "="*60)
    print("TESTING PHASE")
    print("="*60)
    
    # Create testing momentum space grid
    kx_list_test = get_k_grid(n_k_points_test)
    ky_list_test = get_k_grid(n_k_points_test)
    
    # Run simulation using helper function
    return run_simulation_on_grid(kx_list_test, ky_list_test, optimized_strength_durations, n_cycles_test, "Testing", plot=False)

def plot_energy_density_vs_cycles(kx_list, ky_list, strength_durations, max_cycles=20):
    """
    Plot energy density as a function of the number of cooling cycles
    
    Args:
        kx_list, ky_list: momentum space grids
        strength_durations: circuit parameters
        max_cycles: maximum number of cycles to test
    """
    print(f"\nAnalyzing energy density vs cycles (up to {max_cycles} cycles)...")
    
    # Use helper function to run simulation once with max_cycles
    print(f"  Running simulation with {max_cycles} cycles to get energy evolution...")
    E_diff_all_cycles, _ = run_simulation_over_momentum_grid(kx_list, ky_list, strength_durations, max_cycles, verbose=False)
    
    # Calculate energy densities for each cycle count
    cycle_counts = range(1, max_cycles + 1)
    energy_densities = []
    
    for n_cycles in cycle_counts:
        # Use the pre-computed energy differences for this cycle count
        E_diff = E_diff_all_cycles[n_cycles - 1, :, :]  # n_cycles-1 because array is 0-indexed
        energy_density = np.nanmean(E_diff) / 2  # Divide by 2 because we count k and -k together
        energy_densities.append(energy_density)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(cycle_counts, energy_densities, 'b-o', linewidth=2, markersize=6)
    plt.xlabel('Number of Cooling Cycles')
    plt.ylabel('Energy Density')
    plt.title('Energy Density vs Number of Cooling Cycles (Test Data)')
    plt.grid(True, alpha=0.3)
    
    # Add horizontal line at the final value for reference
    plt.axhline(y=energy_densities[-1], color='r', linestyle='--', alpha=0.7, 
                label=f'Final value: {energy_densities[-1]:.6f}')
    
    # Add vertical line at the training cycles for reference
    plt.axvline(x=n_cycles_train, color='g', linestyle=':', alpha=0.7, 
                label=f'Training cycles: {n_cycles_train}')
    
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print(f"\nEnergy density statistics:")
    print(f"  After 1 cycle: {energy_densities[0]:.6f}")
    print(f"  After {n_cycles_train} cycles (training): {energy_densities[n_cycles_train-1]:.6f}")
    print(f"  After {max_cycles} cycles (final): {energy_densities[-1]:.6f}")
    print(f"  Improvement from 1 to {max_cycles} cycles: {energy_densities[0] - energy_densities[-1]:.6f}")
    
    return cycle_counts, energy_densities

def main():
    """Main function to run the variational circuit simulation with training and testing"""
    print("Starting variational circuit simulation with training and testing...")
    print(f"Using {p} layers for variational circuit")
    print(f"Training with {n_cycles_train} cooling cycles")
    print(f"Testing with {n_cycles_test} cooling cycles")
    
    # Phase 1: Training
    optimized_strength_durations, opt_result = train_variational_circuit()
    
    # Phase 2: Testing
    E_diff, single_particle_dm, total_chern_number, system_chern_number, bath_chern_number, energy_density = test_variational_circuit(optimized_strength_durations)

    # Phase 3: Energy density vs cycles analysis
    print("\n" + "="*60)
    print("ENERGY DENSITY VS CYCLES ANALYSIS")
    print("="*60)
    
    # Create testing momentum space grid for the analysis
    kx_list_test = get_k_grid(n_k_points_test)
    ky_list_test = get_k_grid(n_k_points_test)
    
    # Plot energy density vs cycles
    cycle_counts, energy_densities = plot_energy_density_vs_cycles(
        kx_list_test, ky_list_test, optimized_strength_durations, max_cycles=n_cycles_test
    )

    # Plot and print results using helper functions
    print_results(energy_density, total_chern_number, system_chern_number, 
                bath_chern_number, opt_result=opt_result, 
                grid_size=f"{n_k_points_train}x{n_k_points_train} (train, {n_cycles_train} cycles), {n_k_points_test}x{n_k_points_test} (test, {n_cycles_test} cycles)", 
                phase_name="FINAL RESULTS")
    # Use final cycle for plotting
    E_diff_plot = E_diff[-1, :, :] if E_diff.ndim == 3 else E_diff
    plot_results(E_diff_plot, optimized_strength_durations, n_k_points_test, "Test ", show_training_points=True)
    
if __name__ == "__main__":
        main()

