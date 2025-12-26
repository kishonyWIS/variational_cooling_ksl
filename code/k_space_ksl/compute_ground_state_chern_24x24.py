"""
Compute Chern number in the ground state of the 24×24 KSL model.

This script computes the Chern number for the system modes (c^z fermions, modes 0-7)
in the ground state of the 24×24 KSL Hamiltonian on a grid of k-points.
"""

import numpy as np
import sys
import os
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
# Add code directory to path for interger_chern_number
code_dir = os.path.join(os.path.dirname(__file__), '..')
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from ksl_24x24_model import create_KSL_24x24_hamiltonian
from progressive_circuit_expansion_training_24x24 import get_k_grid
from interger_chern_number import chern_from_mixed_spdm, get_chern_number_from_single_particle_dm, chern_from_spdm_with_threshold_eigenvalues, chern_fhs_from_spdm


def compute_ground_state_chern(kx_list, ky_list, Jx=1.0, Jy=1.0, Jz=1.0, kappa=1.0, 
                                verbose=False):
    """
    Compute Chern number from ground state density matrices on a k-point grid.
    
    Args:
        kx_list, ky_list: Lists of momentum values
        Jx, Jy, Jz, kappa: Kitaev parameters
        verbose: Whether to print progress
    
    Returns:
        tuple: (system_chern, total_chern, single_particle_dm)
            - system_chern: Chern number for system modes (0-7, c^z fermions)
            - total_chern: Chern number for all modes (0-23)
            - single_particle_dm: Array of shape (n_kx, n_ky, 24, 24) with ground state density matrices
    """
    n_kx = len(kx_list)
    n_ky = len(ky_list)
    
    # Initialize output array for density matrices
    single_particle_dm = np.zeros((n_kx, n_ky, 24, 24), dtype=complex)
    
    total_points = n_kx * n_ky
    point_count = 0
    
    if verbose:
        print(f"Computing ground states for {total_points} k-points...")
    
    for i_kx, kx in enumerate(kx_list):
        for i_ky, ky in enumerate(ky_list):
            if verbose and point_count % 10 == 0:
                print(f"  Processing k-point {point_count+1}/{total_points}: kx={kx:.4f}, ky={ky:.4f}")
            
            # Create system Hamiltonian (g=0, B=0 for ground state)
            system_hamiltonian = create_KSL_24x24_hamiltonian(
                kx, ky, Jx=Jx, Jy=Jy, Jz=Jz, kappa=kappa, g=0.0, B=0.0
            )
            
            # Get ground state density matrix
            ground_state_dm = system_hamiltonian.get_ground_state()
            
            # Store the full 24×24 density matrix
            single_particle_dm[i_kx, i_ky, :, :] = ground_state_dm
            
            point_count += 1
    
    # Compute Chern numbers
    # System modes: 0-7 (c^z fermions)
    system_chern = chern_from_spdm_with_threshold_eigenvalues(single_particle_dm[:, :, :8, :8])
    
    # Total Chern number: all 24 modes
    total_chern = chern_from_spdm_with_threshold_eigenvalues(single_particle_dm)
    
    if verbose:
        print(f"\nChern number (system modes 0-7): {system_chern:.6f}")
        print(f"Chern number (all modes 0-23): {total_chern:.6f}")

    # Compute Chern number from single-particle density matrix
    system_chern_from_single_particle_dm = get_chern_number_from_single_particle_dm(single_particle_dm[:, :, :8, :8])
    total_chern_from_single_particle_dm = get_chern_number_from_single_particle_dm(single_particle_dm)

    if verbose:
        print(f"Chern number (system modes 0-7) from single-particle density matrix: {system_chern_from_single_particle_dm:.6f}")
        print(f"Chern number (all modes 0-23) from single-particle density matrix: {total_chern_from_single_particle_dm:.6f}")
    
    # Compute Chern number using FHS method
    system_chern_fhs, system_diag = chern_fhs_from_spdm(single_particle_dm[:, :, :8, :8], return_diagnostics=True)
    total_chern_fhs, total_diag = chern_fhs_from_spdm(single_particle_dm, return_diagnostics=True)

    if verbose:
        print(f"Chern number (system modes 0-7) FHS method: {system_chern_fhs:.6f} (n_occ={system_diag['n_occ']}, min_gap={system_diag['min_gap']:.4f})")
        print(f"Chern number (all modes 0-23) FHS method: {total_chern_fhs:.6f} (n_occ={total_diag['n_occ']}, min_gap={total_diag['min_gap']:.4f})")
    
    return (system_chern, total_chern, single_particle_dm, 
            system_chern_from_single_particle_dm, total_chern_from_single_particle_dm,
            system_chern_fhs, total_chern_fhs, system_diag, total_diag)


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description='Compute Chern number in ground state of 24×24 KSL model'
    )
    parser.add_argument('--n-k-points', type=int, default=120,
                       help='Number of k-points per dimension (default: 60)')
    parser.add_argument('--Jx', type=float, default=1.0,
                       help='Jx parameter (default: 1.0)')
    parser.add_argument('--Jy', type=float, default=1.0,
                       help='Jy parameter (default: 1.0)')
    parser.add_argument('--Jz', type=float, default=1.0,
                       help='Jz parameter (default: 1.0)')
    parser.add_argument('--kappa', type=float, default=1.0,
                       help='kappa parameter (default: 1.0)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file to save results (optional)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print progress information')
    
    args = parser.parse_args()
    
    # Create k-point grid
    kx_list = get_k_grid(args.n_k_points)
    ky_list = get_k_grid(args.n_k_points)
    
    print("="*60)
    print("COMPUTING CHERN NUMBER IN GROUND STATE (24×24 MODEL)")
    print("="*60)
    print(f"Grid size: {args.n_k_points}×{args.n_k_points}")
    print(f"Kitaev parameters: Jx={args.Jx}, Jy={args.Jy}, Jz={args.Jz}, kappa={args.kappa}")
    print("="*60)
    
    # Compute Chern numbers
    (system_chern, total_chern, single_particle_dm, 
     system_chern_from_single_particle_dm, total_chern_from_single_particle_dm,
     system_chern_fhs, total_chern_fhs, system_diag, total_diag) = compute_ground_state_chern(
        kx_list, ky_list, Jx=args.Jx, Jy=args.Jy, Jz=args.Jz, kappa=args.kappa,
        verbose=args.verbose
    )
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"System Chern number (modes 0-7, c^z): {system_chern:.8f} (threshold eigenvalues)")
    print(f"Total Chern number (modes 0-23): {total_chern:.8f} (threshold eigenvalues)")
    print(f"System Chern number (modes 0-7, c^z): {system_chern_from_single_particle_dm:.8f} (derivative method)")
    print(f"Total Chern number (modes 0-23): {total_chern_from_single_particle_dm:.8f} (derivative method)")
    print(f"System Chern number (modes 0-7, c^z): {system_chern_fhs:.8f} (FHS method, n_occ={system_diag['n_occ']}, min_gap={system_diag['min_gap']:.4f})")
    print(f"Total Chern number (modes 0-23): {total_chern_fhs:.8f} (FHS method, n_occ={total_diag['n_occ']}, min_gap={total_diag['min_gap']:.4f})")
    print("="*60)


if __name__ == "__main__":
    main()

