"""
Test trotterized cooling using the self-contained KSL model.

This script tests whether a trotterized circuit with many small steps using
time-dependent g(t) and B(t) can cool the system to a low energy state.
"""

import sys
import os
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from k_space_KSL_model import create_KSL_hamiltonian, VariationalCircuit, Hamiltonian, KSLSingleParticleDensityMatrix
from time_dependence_functions import get_g, get_B
from k_space_KSL_model import get_Delta_without_kappa


def create_trotterized_circuit(kx: float, ky: float, num_steps: int = 100, 
                                T: float = 1.0, g0: float = 1.0, B0: float = 0.0, B1: float = 1.0,
                                Jx: float = 1.0, Jy: float = 1.0, 
                                Jz: float = 1.0, kappa: float = 1.0):
    """
    Create a trotterized circuit with time-dependent g(t) and B(t).
    
    Args:
        kx, ky: Momentum values
        num_steps: Number of Trotter steps
        T: Total evolution time
        g0: Maximum g value
        B0: Initial B value
        B1: Final B value
        t1: Time parameter for g(t) bump
        Jx, Jy, Jz, kappa: Kitaev parameters
    
    Returns:
        VariationalCircuit instance
    """
    # Create Hamiltonian (g and B will be time-dependent in the circuit)
    hamiltonian = create_KSL_hamiltonian(kx, ky, Jx=Jx, Jy=Jy, Jz=Jz, kappa=kappa, g=1., B=1.) # g and B are constants for now to ensure the terms are added to the Hamiltonian
    
    # Time step for trotterization
    dt = T / num_steps

    Delta = kappa * get_Delta_without_kappa(kx, ky)
    
    # Initialize parameters - all terms get dt per step
    # For g and B, we'll multiply by the time-dependent values
    parameters = {
        'Jx': np.ones(num_steps) * Jx*dt,
        'Jy': np.ones(num_steps) * Jy*dt,
        'Jz': np.ones(num_steps) * Jz*dt,
        'Delta_A': np.ones(num_steps) * Delta*dt,
        'Delta_B': np.ones(num_steps) * Delta*dt,
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
    
    # Create circuit
    circuit = VariationalCircuit(hamiltonian, parameters)
    
    return circuit


def test_trotterized_cooling(kx: float = 0.5, ky: float = 0.7, num_steps: int = 100, num_cycles: int = 5, Jx: float = 1.0, Jy: float = 1.0, Jz: float = 1.0, kappa: float = 1.0):
    """
    Test trotterized cooling protocol with multiple cycles.
    
    Args:
        kx, ky: Momentum values to test
        num_steps: Number of Trotter steps per cycle
        num_cycles: Number of cooling cycles to run
    """
    print("="*80)
    print("Testing Trotterized Cooling Protocol")
    print("="*80)
    print(f"Momentum: kx={kx:.4f}, ky={ky:.4f}")
    print(f"Number of Trotter steps per cycle: {num_steps}")
    print(f"Number of cooling cycles: {num_cycles}")
    print()
    
    # Create system Hamiltonian for energy computation and ground state
    # (without g and B terms, as energy is measured with respect to system only)
    system_hamiltonian_with_B = create_KSL_hamiltonian(kx, ky, Jx=Jx, Jy=Jy, Jz=Jz, kappa=kappa, g=0.0, B=1.0)
    system_hamiltonian = create_KSL_hamiltonian(kx, ky, Jx=Jx, Jy=Jy, Jz=Jz, kappa=kappa, g=0.0, B=0.0)
    ground_state_matrix_with_B = system_hamiltonian_with_B.get_ground_state()
    ground_state_with_B = KSLSingleParticleDensityMatrix(matrix=ground_state_matrix_with_B)
    print('ground state matrix with B')
    print(ground_state_matrix_with_B)
    #reset all tau
    ground_state_with_B.reset_all_tau()
    print('ground state matrix with B after reset')
    print(ground_state_with_B.matrix)


    # Get ground state
    ground_state_matrix = system_hamiltonian.get_ground_state()
    ground_state = KSLSingleParticleDensityMatrix(matrix=ground_state_matrix)
    E_gs = system_hamiltonian.compute_energy(ground_state.matrix)
    print(f"Ground state energy: {E_gs:.6f}")
    print()
    
    state = KSLSingleParticleDensityMatrix(system_hamiltonian.system_size)
    state.initialize('random')
    state.reset_all_tau()
    E_initial = system_hamiltonian.compute_energy(state.matrix)
    print(f"Initial state energy: {E_initial:.6f}")
    print(f"Energy above ground state: {E_initial - E_gs:.6f}")
    print()
    
    # Create trotterized circuit
    print("Creating trotterized circuit...")
    circuit = create_trotterized_circuit(kx, ky, num_steps=num_steps, T=100., g0 = 0.5, B1=0., B0=7.,
                                         Jx=Jx, Jy=Jy, Jz=Jz, kappa=kappa)
    
    # Get the full unitary
    print("Computing circuit unitary...")
    Ud = circuit.get_unitary()
    print('Ud =', Ud)
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
        
        if cycle == 0:
            print(f"{cycle+1:6d} {E_current:15.6f} {E_above_gs:15.6f}")
        else:
            print(f"{cycle+1:6d} {E_current:15.6f} {E_above_gs:15.6f}")
    
    print("-"*80)
    print()

    print('ground state matrix')
    print(ground_state_matrix)

    print('final state matrix')
    print(state.matrix)


if __name__ == "__main__":
    kx = 0.6
    ky = 1.5
    test_trotterized_cooling(kx=kx, ky=ky, num_steps=10000, num_cycles=5, Jx=1.00, Jy=1.00, Jz=1.00, kappa=1.00)