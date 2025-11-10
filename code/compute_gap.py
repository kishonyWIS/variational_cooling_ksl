"""
Compute the energy gap of the KSL model for a given grid of k points.

The gap is defined as the minimal absolute value of the ground state energy
over the momentum grid points.
"""

import numpy as np
import sys
import os

# Add project root to path to access root-level dependencies
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from translational_invariant_KSL import get_KSL_model, get_Delta, get_f

# Model parameters
kappa = 1.
Jx = 1.
Jy = 1.
Jz = 1.

# For gap computation, we use the static Hamiltonian (g=0, B=0)
# The ground state energy is computed from the static part (f and Delta)
g_static = 0.0
B_static = 0.0

num_cooling_sublattices = 2


def compute_gap(kx_list, ky_list, verbose=True):
    """
    Compute the energy gap for a given grid of k points.
    
    The gap is defined as min(|E_gs|) over all momentum points.
    
    Args:
        kx_list: array of kx values
        ky_list: array of ky values
        verbose: whether to print progress
        
    Returns:
        gap: the energy gap (min(|E_gs|))
        min_E_gs: the minimum ground state energy (before taking absolute value)
        min_kx: kx value at which minimum occurs
        min_ky: ky value at which minimum occurs
    """
    E_gs_values = []
    kx_min = None
    ky_min = None
    min_E_gs = float('inf')
    
    n_total = len(kx_list) * len(ky_list)
    n_computed = 0
    
    for i_kx, kx in enumerate(kx_list):
        for i_ky, ky in enumerate(ky_list):
            # Get f and Delta for this k point
            f = get_f(kx, ky, Jx, Jy, Jz)
            Delta = get_Delta(kx, ky, kappa)
            
            # Get the KSL model with static g=0, B=0 to compute ground state energy
            # This gives us the ground state energy of the static Hamiltonian
            hamiltonian, S, E_gs = get_KSL_model(
                f=f, Delta=Delta, g=g_static, B=B_static,
                initial_state='ground',  # Use ground state initialization
                num_cooling_sublattices=num_cooling_sublattices
            )
            
            E_gs_values.append(E_gs)
            
            # Track minimum absolute value
            if abs(E_gs) < abs(min_E_gs):
                min_E_gs = E_gs
                kx_min = kx
                ky_min = ky
            
            n_computed += 1
            if verbose and n_computed % 100 == 0:
                print(f"Progress: {n_computed}/{n_total} k-points computed ({100*n_computed/n_total:.1f}%)")
    
    # Compute gap as min(|E_gs|)
    gap = abs(min_E_gs)
    
    return gap, min_E_gs, kx_min, ky_min


def main():
    """Compute the gap on the testing grid"""
    print("="*60)
    print("Computing Energy Gap of KSL Model")
    print("="*60)
    
    # Test grid: same as used in variational_circuit_KSL_numba.py
    n_k_points_test = 1 + 6 * 20  # 121 points
    print(f"\nUsing test grid: {n_k_points_test}x{n_k_points_test} = {n_k_points_test**2} points")
    
    # Create momentum grid
    kx_list = np.linspace(-np.pi, np.pi, n_k_points_test)
    ky_list = np.linspace(-np.pi, np.pi, n_k_points_test)
    
    print(f"kx range: [{kx_list[0]:.4f}, {kx_list[-1]:.4f}]")
    print(f"ky range: [{ky_list[0]:.4f}, {ky_list[-1]:.4f}]")
    print(f"\nModel parameters:")
    print(f"  Jx = {Jx}, Jy = {Jy}, Jz = {Jz}")
    print(f"  kappa = {kappa}")
    print("\nComputing ground state energies...")
    
    # Compute gap
    gap, min_E_gs, kx_min, ky_min = compute_gap(kx_list, ky_list, verbose=True)
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Energy gap = min(|E_gs|) = {gap:.6f}")
    print(f"Minimum ground state energy: {min_E_gs:.6f}")
    print(f"Location of minimum: kx = {kx_min:.6f}, ky = {ky_min:.6f}")
    print("="*60)
    
    return gap, min_E_gs, kx_min, ky_min


if __name__ == "__main__":
    gap, min_E_gs, kx_min, ky_min = main()

