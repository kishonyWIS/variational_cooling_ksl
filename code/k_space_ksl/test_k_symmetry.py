"""
Test that the energy density reached by the cooling circuit is the same at k and -k
for any Hamiltonian parameters and variational parameters.
"""

import sys
import os
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import from the same directory
from progressive_circuit_expansion_training import (
    simulate_single_kpoint
)
from k_space_KSL_model import create_KSL_hamiltonian, VariationalCircuit, KSLSingleParticleDensityMatrix


def test_k_symmetry(num_tests=100, n_cycles=5, p=5, verbose=True):
    """
    Test that energy density at (kx, ky) equals energy density at (-kx, -ky)
    for random Hamiltonian parameters and variational parameters.
    
    Args:
        num_tests: Number of random test cases
        n_cycles: Number of cooling cycles
        p: Number of circuit layers
        verbose: Whether to print details
    
    Returns:
        bool: True if all tests pass
    """
    print("="*80)
    print("Testing k -> -k Symmetry")
    print("="*80)
    print(f"Running {num_tests} random test cases...")
    print(f"Circuit layers: {p}, Cooling cycles: {n_cycles}")
    print()
    
    np.random.seed(42)  # For reproducibility
    failures = []
    
    for test_idx in range(num_tests):
        # Generate random k point
        kx = np.random.uniform(-np.pi, np.pi)
        ky = np.random.uniform(-np.pi, np.pi)
        
        # Generate random Hamiltonian parameters
        Jx = np.random.uniform(-2.0, 2.0)
        Jy = np.random.uniform(-2.0, 2.0)
        Jz = np.random.uniform(-2.0, 2.0)
        kappa = np.random.uniform(-2.0, 2.0)  # Keep kappa positive
        
        # Generate random variational parameters (old 6-parameter structure)
        # Allow negative values to test symmetry with negative parameters
        parameters = {
            'Jx': np.random.uniform(-1.0, 1.0, size=p),
            'Jy': np.random.uniform(-1.0, 1.0, size=p),
            'Jz': np.random.uniform(-1.0, 1.0, size=p),
            'kappa': np.random.uniform(-1.0, 1.0, size=p),
            'g': np.random.uniform(-1.0, 1.0, size=p),
            'B': np.random.uniform(-1.0, 1.0, size=p),
        }
        
        # Simulate at k
        _, E_diff_list_k, E_gs_k, _ = simulate_single_kpoint(
            kx, ky, parameters, n_cycles, Jx, Jy, Jz, kappa
        )
        E_final_k = E_diff_list_k[-1] if len(E_diff_list_k) > 0 else 0.0
        
        # Simulate at -k
        _, E_diff_list_negk, E_gs_negk, _ = simulate_single_kpoint(
            -kx, -ky, parameters, n_cycles, Jx, Jy, Jz, kappa
        )
        E_final_negk = E_diff_list_negk[-1] if len(E_diff_list_negk) > 0 else 0.0
        
        # Check symmetry
        energy_diff = abs(E_final_k - E_final_negk)
        tolerance = 1e-6
        
        if energy_diff > tolerance:
            failures.append({
                'test_idx': test_idx,
                'kx': kx,
                'ky': ky,
                'Jx': Jx,
                'Jy': Jy,
                'Jz': Jz,
                'kappa': kappa,
                'E_k': E_final_k,
                'E_negk': E_final_negk,
                'diff': energy_diff
            })
            
            if verbose:
                print(f"✗ Test {test_idx+1} FAILED:")
                print(f"  k=({kx:.4f}, {ky:.4f}), -k=({-kx:.4f}, {-ky:.4f})")
                print(f"  Jx={Jx:.4f}, Jy={Jy:.4f}, Jz={Jz:.4f}, kappa={kappa:.4f}")
                print(f"  E(k) = {E_final_k:.8f}")
                print(f"  E(-k) = {E_final_negk:.8f}")
                print(f"  Difference = {energy_diff:.8f}")
                print()
        elif verbose and (test_idx + 1) % 10 == 0:
            print(f"✓ Test {test_idx+1}/{num_tests} passed")
    
    # Summary
    print("="*80)
    if len(failures) == 0:
        print(f"✓ All {num_tests} tests PASSED!")
        print("Symmetry k -> -k is preserved.")
        return True
    else:
        print(f"✗ {len(failures)}/{num_tests} tests FAILED")
        print("\nFirst few failures:")
        for i, failure in enumerate(failures[:5]):
            print(f"\nFailure {i+1}:")
            print(f"  k=({failure['kx']:.4f}, {failure['ky']:.4f})")
            print(f"  Jx={failure['Jx']:.4f}, Jy={failure['Jy']:.4f}, Jz={failure['Jz']:.4f}, kappa={failure['kappa']:.4f}")
            print(f"  E(k) = {failure['E_k']:.8f}")
            print(f"  E(-k) = {failure['E_negk']:.8f}")
            print(f"  Difference = {failure['diff']:.8f}")
        return False


if __name__ == "__main__":
    success = test_k_symmetry(num_tests=100, n_cycles=5, p=5, verbose=True)
    sys.exit(0 if success else 1)

