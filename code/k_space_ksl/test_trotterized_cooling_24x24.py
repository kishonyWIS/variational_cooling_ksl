"""
Test trotterized cooling using the 24×24 KSL model.

This script tests whether a trotterized circuit with many small steps using
time-dependent g(t) and B(t) can cool the system to a low energy state using
the 24×24 model with supercell decomposition.
"""

import sys
import os
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ksl_24x24_model import (
    create_KSL_24x24_hamiltonian, 
    KSL24SingleParticleDensityMatrix,
    convert_k_to_K,
    get_all_kappa_term_names
)
from hamiltonian_base import VariationalCircuit
from time_dependence_functions import get_g, get_B


def create_trotterized_parameters_24x24(num_steps, T, g0, B0, B1):
    """
    Create trotterized evolution parameters for 24×24 model.
    
    Args:
        num_steps: Number of Trotter steps
        T: Total evolution time
        g0, B0, B1: Time-dependent function parameters
    
    Returns:
        Dictionary with all parameter names. All kappa terms get the same value.
    """
    # Time step for trotterization
    dt = T / num_steps
    
    # Initialize parameters
    parameters = {}
    
    # J terms
    parameters['Jx_r0'] = np.ones(num_steps) * dt
    parameters['Jx_r1'] = np.ones(num_steps) * dt
    parameters['Jx_r2'] = np.ones(num_steps) * dt
    parameters['Jx_r3'] = np.ones(num_steps) * dt
    
    parameters['Jy_A1_B0'] = np.ones(num_steps) * dt
    parameters['Jy_A0_B1'] = np.ones(num_steps) * dt
    parameters['Jy_A3_B2'] = np.ones(num_steps) * dt
    parameters['Jy_A2_B3'] = np.ones(num_steps) * dt
    
    parameters['Jz_A2_B0'] = np.ones(num_steps) * dt
    parameters['Jz_A0_B2'] = np.ones(num_steps) * dt
    parameters['Jz_A3_B1'] = np.ones(num_steps) * dt
    parameters['Jz_A1_B3'] = np.ones(num_steps) * dt
    
    # Kappa terms: all 24 get the same value (dt)
    kappa_names = get_all_kappa_term_names()
    for name in kappa_names:
        parameters[name] = np.ones(num_steps) * dt
    
    # g terms: for all 4 offsets
    for r in range(4):
        parameters[f'g_A_r{r}'] = np.ones(num_steps)
        parameters[f'g_B_r{r}'] = np.ones(num_steps)
    
    # B terms: for all 4 offsets
    for r in range(4):
        parameters[f'B_A_r{r}'] = np.ones(num_steps)
        parameters[f'B_B_r{r}'] = np.ones(num_steps)
    
    # Apply time-dependent scaling for g and B terms
    for step in range(num_steps):
        t = step * dt
        g_t = get_g(t, g0, T, T/4)
        B_t = get_B(t, B0, B1, T)
        
        # g and B parameters are the time-dependent values times dt
        for r in range(4):
            parameters[f'g_A_r{r}'][step] = g_t * dt
            parameters[f'g_B_r{r}'][step] = g_t * dt
            parameters[f'B_A_r{r}'][step] = B_t * dt
            parameters[f'B_B_r{r}'][step] = B_t * dt
    
    return parameters


def create_trotterized_circuit_24x24(kx: float, ky: float, num_steps: int = 100, 
                                     T: float = 1.0, g0: float = 1.0, B0: float = 0.0, B1: float = 1.0,
                                     Jx: float = 1.0, Jy: float = 1.0, 
                                     Jz: float = 1.0, kappa: float = 1.0):
    """
    Create a trotterized circuit with time-dependent g(t) and B(t) for 24×24 model.
    
    Args:
        kx, ky: Momentum values (will be converted to K_i, K_j)
        num_steps: Number of Trotter steps
        T: Total evolution time
        g0: Maximum g value
        B0: Initial B value
        B1: Final B value
        Jx, Jy, Jz, kappa: Kitaev parameters
    
    Returns:
        VariationalCircuit instance
    """
    # Convert k-space to K-space
    K_i, K_j = convert_k_to_K(kx, ky)
    
    # Create Hamiltonian (g and B will be time-dependent in the circuit)
    # g and B are constants for now to ensure the terms are added to the Hamiltonian
    hamiltonian = create_KSL_24x24_hamiltonian(K_i, K_j, Jx=Jx, Jy=Jy, Jz=Jz, kappa=kappa, g=1., B=1.)
    
    # Get parameters from helper function
    parameters = create_trotterized_parameters_24x24(num_steps, T, g0, B0, B1)
    
    # Create circuit
    circuit = VariationalCircuit(hamiltonian, parameters)
    
    return circuit


def test_trotterized_cooling_24x24(kx: float = 0.5, ky: float = 0.7, num_steps: int = 100, 
                                   num_cycles: int = 5, Jx: float = 1.0, Jy: float = 1.0, 
                                   Jz: float = 1.0, kappa: float = 1.0):
    """
    Test trotterized cooling protocol with multiple cycles for 24×24 model.
    
    Args:
        kx, ky: Momentum values to test
        num_steps: Number of Trotter steps per cycle
        num_cycles: Number of cooling cycles to run
    """
    print("="*80)
    print("Testing Trotterized Cooling Protocol (24×24 Model)")
    print("="*80)
    print(f"Momentum: kx={kx:.4f}, ky={ky:.4f}")
    K_i, K_j = convert_k_to_K(kx, ky)
    print(f"Bloch wavevector: K_i={K_i:.4f}, K_j={K_j:.4f}")
    print(f"Number of Trotter steps per cycle: {num_steps}")
    print(f"Number of cooling cycles: {num_cycles}")
    print()
    
    # Create system Hamiltonian for energy computation and ground state
    # (without g and B terms, as energy is measured with respect to system only)
    system_hamiltonian = create_KSL_24x24_hamiltonian(K_i, K_j, Jx=Jx, Jy=Jy, Jz=Jz, kappa=kappa, g=0.0, B=0.0)

    # Get ground state
    ground_state_matrix = system_hamiltonian.get_ground_state()
    ground_state = KSL24SingleParticleDensityMatrix(matrix=ground_state_matrix)
    E_gs = system_hamiltonian.compute_energy(ground_state.matrix)
    print(f"Ground state energy: {E_gs:.6f}")
    print()
    
    state = KSL24SingleParticleDensityMatrix(system_hamiltonian.system_size)
    state.initialize('random')
    state.reset_all_tau()
    E_initial = system_hamiltonian.compute_energy(state.matrix)
    print(f"Initial state energy: {E_initial:.6f}")
    print(f"Energy above ground state: {E_initial - E_gs:.6f}")
    print()
    
    # Create trotterized circuit
    print("Creating trotterized circuit...")
    circuit = create_trotterized_circuit_24x24(kx, ky, num_steps=num_steps, T=100., g0=0.5, B1=0., B0=7.,
                                               Jx=Jx, Jy=Jy, Jz=Jz, kappa=kappa)
    
    # Get the full unitary
    print("Computing circuit unitary...")
    Ud = circuit.get_unitary()
    print(f"Unitary shape: {Ud.shape}")
    print(f"Is unitary: {np.allclose(Ud @ Ud.conj().T, np.eye(system_hamiltonian.system_size))}")
    print()
    
    # Run multiple cooling cycles
    print("Running cooling cycles...")
    print("-"*80)
    print(f"{'Cycle':>6} {'Energy':>15} {'E - E_gs':>15}")
    print("-"*80)
    
    for cycle in range(num_cycles):
        
        # Apply the circuit unitary
        state.evolve_state_with_unitary(Ud)

        # Reset tau after each cycle
        state.reset_all_tau()
        
        # Compute and print energy (using system Hamiltonian only)
        E_current = system_hamiltonian.compute_energy(state.matrix)
        E_above_gs = E_current - E_gs
        
        print(f"{cycle+1:6d} {E_current:15.6f} {E_above_gs:15.6f}")
    
    print("-"*80)
    print()
    
    print('Ground state energy:', E_gs)
    print('Final state energy:', E_current)
    print('Final energy above ground state:', E_above_gs)


if __name__ == "__main__":
    kx = 0.6
    ky = 0.9
    test_trotterized_cooling_24x24(kx=kx, ky=ky, num_steps=1000, num_cycles=5, 
                                    Jx=1.00, Jy=1.00, Jz=1.00, kappa=1.00)

