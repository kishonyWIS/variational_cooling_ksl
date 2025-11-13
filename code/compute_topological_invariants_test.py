"""
Simple script to compute topological invariants from steady state.

Computes both the Chern number and the plaquette phase sum
from the single-particle density matrix after 40 cooling cycles
for saved circuit parameters with p=5, res=3.
"""

import numpy as np
import sys
import os
from interger_chern_number import chern_from_mixed_spdm

# Add parent directory to path to import from variational_circuit_KSL_numba
sys.path.insert(0, os.path.dirname(__file__))

from variational_circuit_KSL_numba import (
    load_optimized_parameters_for_res_p,
    simulate_grid,
    get_chern_number_from_single_particle_dm,
    get_k_grid,
)

# Parameters
res = 3
p = 10
n_cycles = 40

# Create k-grid
n_k_points = 6 * 10
kx_list = get_k_grid(n_k_points)
ky_list = get_k_grid(n_k_points)

print("="*60)
print("Computing Topological Invariants")
print("="*60)
print(f"Parameters: res={res}, p={p}, n_cycles={n_cycles}")
print(f"Grid size: {n_k_points + 1}x{n_k_points + 1}")
print()

# Load optimized parameters
print("Loading optimized parameters...")
try:
    strength_durations = load_optimized_parameters_for_res_p(res, p)
    print(f"Successfully loaded parameters for res={res}, p={p}")
except FileNotFoundError as e:
    print(f"Error: {e}")
    sys.exit(1)

# Set global p (needed for some internal functions)
import variational_circuit_KSL_numba as vc_module
vc_module.p = p

# Run simulation to get steady state
print(f"\nRunning simulation with {n_cycles} cycles to reach steady state...")
E_diff, single_particle_dm = simulate_grid(kx_list, ky_list, strength_durations, n_cycles=n_cycles, verbose=True)

# Extract final cycle energy if multiple cycles
if E_diff.ndim == 3:
    energy_density = np.nanmean(E_diff[-1, :, :]) / 2
else:
    energy_density = np.nanmean(E_diff) / 2

print(f"\nEnergy density: {energy_density:.6f}")

# Compute topological invariants
print("\nComputing topological invariants...")
print("-" * 60)


chern_number = get_chern_number_from_single_particle_dm(single_particle_dm[:,:,:2,:2])
print(f"Chern number: {chern_number:.6f}")

# Chern number from exact method
chern_number_exact = chern_from_mixed_spdm(single_particle_dm[:,:,:2,:2])
print(f"Chern number (exact): {chern_number_exact:.6f}")

