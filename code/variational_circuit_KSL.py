from itertools import product
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize
import sys
import os

# Add project root to path to access root-level dependencies
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from time_dependence_functions import get_g, get_B
from translational_invariant_KSL import get_KSL_model, get_Delta, get_f

# Data directory path (relative to this file)
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')

# Parameters from original file
g0 = 0.5
B1 = 0.
B0 = 7.
kappa = 1.
Jx = 1.
Jy = 1.
Jz = 1.

# Variational circuit parameters
T = 50
p = 4  # Number of layers in the variational circuit
n_k_points = 1 + 6*1

integration_params = dict(name='vode', nsteps=6000, rtol=1e-6, atol=1e-10)

smoothed_g = lambda tt: get_g(tt, g0, T, T/4) #lambda tt: 0#
smoothed_B = lambda tt: get_B(tt, B0, B1, T) #lambda tt: 0#

def get_chern_number_from_single_particle_dm(single_particle_dm):
    """Calculate Chern number from single-particle density matrix"""
    dP_dkx = np.diff(single_particle_dm, axis=0)[:,:-1,:,:]
    dP_dky = np.diff(single_particle_dm, axis=1)[:-1,:,:,:]
    P = single_particle_dm[:-1,:-1,:,:]
    integrand = np.zeros(P.shape[0:2],dtype=complex)
    for i_kx, i_ky in product(range(P.shape[0]), repeat=2):
        integrand[i_kx,i_ky] = np.trace(P[i_kx,i_ky,:,:] @ (dP_dkx[i_kx,i_ky,:,:] @ dP_dky[i_kx,i_ky,:,:] - dP_dky[i_kx,i_ky,:,:] @ dP_dkx[i_kx,i_ky,:,:]))
    return (np.sum(integrand)/(2*np.pi)).imag

def get_individual_hamiltonian_terms(kx, ky, t):
    """Get individual Hamiltonian term matrices using get_KSL_model for consistency"""
    Delta = get_Delta(kx, ky, kappa)
    g_t = smoothed_g(t)
    B_t = smoothed_B(t)
    
    H_terms = {}
    
    # 1. Jx term - only Jx contribution to f
    f_Jx = get_f(kx, ky, Jx, 0, 0)  # Only Jx term
    hamiltonian_Jx, _, _ = get_KSL_model(f=f_Jx, Delta=0, g=0, B=0, 
                                        initial_state='product', num_cooling_sublattices=2)
    H_terms['Jx'] = hamiltonian_Jx.get_matrix()
    
    # 2. Jy term - only Jy contribution to f
    f_Jy = get_f(kx, ky, 0, Jy, 0)  # Only Jy term
    hamiltonian_Jy, _, _ = get_KSL_model(f=f_Jy, Delta=0, g=0, B=0, 
                                        initial_state='product', num_cooling_sublattices=2)
    H_terms['Jy'] = hamiltonian_Jy.get_matrix()
    
    # 3. Jz term - only Jz contribution to f
    f_Jz = get_f(kx, ky, 0, 0, Jz)  # Only Jz term
    hamiltonian_Jz, _, _ = get_KSL_model(f=f_Jz, Delta=0, g=0, B=0, 
                                        initial_state='product', num_cooling_sublattices=2)
    H_terms['Jz'] = hamiltonian_Jz.get_matrix()
    
    # 4. kappa term - only Delta contribution
    hamiltonian_kappa, _, _ = get_KSL_model(f=0, Delta=Delta, g=0, B=0, 
                                           initial_state='product', num_cooling_sublattices=2)
    H_terms['kappa'] = hamiltonian_kappa.get_matrix()
    
    # 5. g term - only g coupling contribution
    hamiltonian_g, _, _ = get_KSL_model(f=0, Delta=0, g=g_t, B=0, 
                                       initial_state='product', num_cooling_sublattices=2)
    H_terms['g'] = hamiltonian_g.get_matrix()
    
    # 6. B term - only B coupling contribution
    hamiltonian_B, _, _ = get_KSL_model(f=0, Delta=0, g=0, B=B_t, 
                                       initial_state='product', num_cooling_sublattices=2)
    H_terms['B'] = hamiltonian_B.get_matrix()
    
    return H_terms

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
        # Get time for this layer (distributed across the period T)
        t = layer * T / p
        
        # Get individual Hamiltonian terms at this time
        H_terms = get_individual_hamiltonian_terms(kx, ky, t)
        
        # Apply each term with its combined strength*duration for this layer
        for term_name, H_term in H_terms.items():
            strength_duration = strength_durations[term_name][layer]
            U_term = expm(1j * H_term.conj() * strength_duration / 4)
            Ud = U_term @ Ud
    
    return Ud

def trotterized_evolution_parameters():
    """
    Create trotterized evolution parameters based on the original adiabatic evolution
    This provides a starting point for the variational optimization
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
    return strength_durations

def single_cooling_cycle_variational(kx, ky, strength_durations):
    """
    Perform a single cooling cycle using the variational circuit
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

    # ### !!!
    # Ud_old = hamiltonian.full_cycle_unitary_faster(integration_params, 0, T)

    # #compare Ud and Ud_old up to a global phase by minimizing the distance
    # Ud_Ud_old_distance = minimize(lambda x: np.linalg.norm(Ud * np.exp(1j*x) - Ud_old), x0=np.zeros(6))
    # print('='*100)
    # print('Ud and Ud_old distance = ' + str(Ud_Ud_old_distance.fun))
    # print('='*100)
    # #check if both are unitary
    # print('Ud distance from unitary = ' + str(np.linalg.norm(Ud @ Ud.conj().T - np.eye(6))))
    # print('Ud_old distance from unitary = ' + str(np.linalg.norm(Ud_old @ Ud_old.conj().T - np.eye(6))))
    # print('='*100)
    # # round to 5 decimal places
    # print(np.round(Ud, 4))
    # print('='*100)
    # print(np.round(Ud_old, 4))
    # print('='*100)

    # Apply the variational circuit
    S.evolve_with_unitary(Ud)
    
    # Calculate final energy
    E_final = S.get_energy(hamiltonian.get_matrix(T))
    E_diff = E_final - E_gs
    
    return S, E_diff, E_gs

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

def objective_function(vector, kx_list, ky_list):
    """
    Objective function to minimize: energy density
    """
    # Convert vector back to strength_durations
    strength_durations = vector_to_strength_durations(vector)
    
    # Initialize energy difference array
    E_diff = np.zeros((len(kx_list), len(ky_list)))
    
    # Loop over momentum space
    for i_kx, kx in enumerate(kx_list):
        for i_ky, ky in enumerate(ky_list):
            # Perform single cooling cycle
            S, E_diff_val, E_gs = single_cooling_cycle_variational(kx, ky, strength_durations)
            E_diff[i_kx, i_ky] = E_diff_val
    
    # Calculate average energy density
    energy_density = np.nanmean(E_diff) / 2  # Divide by 2 because we count k and -k together
    
    return energy_density

def optimize_strength_durations(kx_list, ky_list, initial_strength_durations=None, method='L-BFGS-B'):
    """
    Optimize strength_durations to minimize energy density
    
    Args:
        kx_list, ky_list: momentum space grid
        initial_strength_durations: initial guess (if None, uses trotterized evolution)
        method: optimization method ('L-BFGS-B', 'SLSQP', etc.)
    
    Returns:
        optimized_strength_durations: dictionary with optimized parameters
        optimization_result: scipy optimization result
    """
    print("Starting optimization...")
    
    # Get initial guess
    if initial_strength_durations is None:
        initial_strength_durations = trotterized_evolution_parameters()
    
    # Convert to vector
    initial_vector = strength_durations_to_vector(initial_strength_durations)
    
    # Set bounds: allow both positive and negative values for flexibility
    # but keep them reasonable (e.g., -10 to 10)
    bounds = [(-10.0, 10.0)] * len(initial_vector)
    
    print(f"Optimizing {len(initial_vector)} parameters using {method}")
    print(f"Initial energy density: {objective_function(initial_vector, kx_list, ky_list):.6f}")
    
    # Perform optimization
    result = minimize(
        objective_function,
        initial_vector,
        args=(kx_list, ky_list),
        method=method,
        bounds=bounds,
        options={'maxiter': 100, 'disp': True}
    )
    
    # Convert result back to strength_durations
    optimized_strength_durations = vector_to_strength_durations(result.x)
    
    print(f"Optimization completed!")
    print(f"Final energy density: {result.fun:.6f}")
    print(f"Success: {result.success}")
    print(f"Iterations: {result.nit}")
    
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

def test_optimization_small_grid():
    """Test optimization with a very small grid for quick testing"""
    print("Testing optimization with small grid...")
    
    # Use a very small grid for testing
    kx_list = np.linspace(-np.pi, np.pi, 3)  # Only 3 points
    ky_list = np.linspace(-np.pi, np.pi, 3)  # Only 3 points
    
    print(f"Test grid: {len(kx_list)}x{len(ky_list)} = {len(kx_list)*len(ky_list)} points")
    
    # Get initial parameters
    initial_strength_durations = trotterized_evolution_parameters()
    
    # Test objective function with initial parameters
    initial_energy = objective_function(
        strength_durations_to_vector(initial_strength_durations), 
        kx_list, ky_list
    )
    print(f"Initial energy density: {initial_energy:.6f}")
    
    # Run optimization
    optimized_strength_durations, opt_result = optimize_strength_durations(
        kx_list, ky_list, 
        initial_strength_durations=initial_strength_durations,
        method='L-BFGS-B'
    )
    
    print(f"Test completed! Final energy: {opt_result.fun:.6f}")
    return optimized_strength_durations, opt_result

def main():
    """Main function to run the variational circuit simulation with optimization"""
    print("Starting variational circuit simulation with optimization...")
    
    # Create momentum space grid - start with a small grid for testing
    kx_list = np.linspace(-np.pi, np.pi, n_k_points)
    ky_list = np.linspace(-np.pi, np.pi, n_k_points)
    
    print(f"Using {p} layers for variational circuit")
    print(f"Momentum grid: {len(kx_list)}x{len(ky_list)} = {len(kx_list)*len(ky_list)} points")
    
    # Get initial trotterized evolution parameters
    initial_strength_durations = trotterized_evolution_parameters()
    print(f"Initial strength*duration parameters shape: {[(k, v.shape) for k, v in initial_strength_durations.items()]}")
    
    # Optimize the strength_durations
    optimized_strength_durations, opt_result = optimize_strength_durations(
        kx_list, ky_list, 
        initial_strength_durations=initial_strength_durations,
        method='L-BFGS-B'
    )
    
    # Now run the full simulation with optimized parameters
    print("\nRunning full simulation with optimized parameters...")
    
    # Initialize arrays for results
    E_diff = np.zeros((len(kx_list), len(ky_list)))
    single_particle_dm = np.zeros((n_k_points, n_k_points, 6, 6), dtype=complex)
    
    # Loop over momentum space
    for i_kx, kx in enumerate(kx_list):
        print(f'Processing kx={kx:.3f} ({i_kx+1}/{len(kx_list)})')
        for i_ky, ky in enumerate(ky_list):
            # Perform single cooling cycle with optimized parameters
            S, E_diff_val, E_gs = single_cooling_cycle_variational(kx, ky, optimized_strength_durations)
            
            # Store results
            E_diff[i_kx, i_ky] = E_diff_val
            single_particle_dm[i_kx, i_ky, :, :] = S.matrix
    
    # Calculate Chern numbers
    total_chern_number = get_chern_number_from_single_particle_dm(single_particle_dm)
    system_chern_number = get_chern_number_from_single_particle_dm(single_particle_dm[:,:,:2,:2])
    bath_chern_number = get_chern_number_from_single_particle_dm(single_particle_dm[:,:,2:,2:])
    
    # Calculate average energy density
    energy_density = np.nanmean(E_diff) / 2  # Divide by 2 because we count k and -k together

    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(E_diff, extent=[-np.pi, np.pi, -np.pi, np.pi], origin='lower')
    plt.colorbar(label='Energy difference')
    plt.xlabel('kx')
    plt.ylabel('ky')
    plt.title('Energy difference after optimization')
    
    plt.subplot(1, 2, 2)
    # Plot the optimized strength_durations for visualization
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
    
    print(f"\nFinal Results:")
    print(f"Energy density: {energy_density:.6f}")
    print(f"Total Chern number: {total_chern_number:.6f}")
    print(f"System Chern number: {system_chern_number:.6f}")
    print(f"Bath Chern number: {bath_chern_number:.6f}")
    print(f"Optimization success: {opt_result.success}")
    print(f"Optimization iterations: {opt_result.nit}")
    
    # Save optimized parameters
    save_path = os.path.join(DATA_DIR, 'optimized_strength_durations.npz')
    np.savez(save_path, **optimized_strength_durations)
    print(f"Optimized parameters saved to '{save_path}'")
    
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Run test with small grid
        test_optimization_small_grid()
    else:
        # Run full optimization
        main()
