"""
Generate all figures for the LaTeX summary document (24×24 model).

This script loads optimized parameters and generates all plots needed for the paper:
1. energy_vs_cycles.pdf - Energy density vs. number of cooling cycles
2. energy_vs_p_by_res.pdf - Energy density vs. circuit depth p for different training system sizes
3. chern_vs_p.pdf - Chern number vs. circuit depth p
4. energy_heatmap_train.pdf - Energy difference heatmap on training grid
5. energy_heatmap_test.pdf - Energy difference heatmap on test grid
6. parameters_vs_layer.pdf - Optimized parameters vs. layer number
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path to import from progressive_circuit_expansion_training_24x24
sys.path.insert(0, os.path.dirname(__file__))

from progressive_circuit_expansion_training_24x24 import (
    load_optimized_parameters_for_res_p,
    simulate_grid_with_analysis,
    simulate_grid,
    get_k_grid,
    DATA_DIR,
)
from interger_chern_number import chern_fhs_from_spdm
from ksl_24x24_model import create_KSL_24x24_hamiltonian

# Figure output directory
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '../../figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Number of cycles for convergence to steady state (used for all plots)
N_CYCLES_STEADY = 40

# Kitaev parameters (default values)
JX = 1.0
JY = 1.0
JZ = 1.0
KAPPA = 1.0

# Consistent formatting constants
FIG_SIZE = (7, 4)  # Standard figure size for all plots

FONT_SIZE_LABELS = 20
FONT_SIZE_LEGEND = 17
FONT_SIZE_TICKS = 17

# Parameters directory for 24×24 model
PARAMS_DIR_24X24 = os.path.join(DATA_DIR, 'optimized_parameters_24x24')


def plot_energy_vs_cycles(res_val=3, p_val=5, max_cycles=20, Jx=JX, Jy=JY, Jz=JZ, kappa=KAPPA):
    """
    Generate energy_vs_cycles.pdf showing energy density vs. number of cooling cycles
    
    Args:
        res_val: resolution parameter
        p_val: number of layers
        max_cycles: maximum number of cycles to plot
        Jx, Jy, Jz, kappa: Kitaev parameters
    """
    print(f"\n{'='*60}")
    print("Generating energy_vs_cycles.pdf (24×24 model)")
    print(f"{'='*60}")
    
    # Load parameters
    strength_durations = load_optimized_parameters_for_res_p(res_val, p_val, input_dir=PARAMS_DIR_24X24)
    
    # Create grids
    n_k_points_train = 6 * res_val
    n_k_points_test = 6 * 20  # 120
    
    kx_list_train = get_k_grid(n_k_points_train)
    ky_list_train = get_k_grid(n_k_points_train)
    kx_list_test = get_k_grid(n_k_points_test)
    ky_list_test = get_k_grid(n_k_points_test)
    
    # Run simulation for training grid
    print(f"  Running simulation on training grid ({n_k_points_train}x{n_k_points_train}) with {max_cycles} cycles...")
    E_diff_train_all_cycles, _ = simulate_grid(kx_list_train, ky_list_train, strength_durations, 
                                                max_cycles, Jx, Jy, Jz, kappa, verbose=False)
    
    # Run simulation for test grid
    print(f"  Running simulation on test grid ({n_k_points_test}x{n_k_points_test}) with {max_cycles} cycles...")
    E_diff_test_all_cycles, _ = simulate_grid(kx_list_test, ky_list_test, strength_durations, 
                                              max_cycles, Jx, Jy, Jz, kappa, verbose=False)
    
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
    plt.figure(figsize=FIG_SIZE)
    plt.plot(cycle_counts, energy_densities_train, 'b-o', linewidth=2, markersize=6, 
             label=f'Train')
    plt.plot(cycle_counts, energy_densities_test, 'r-s', linewidth=2, markersize=6, 
             label=f'Test')
    
    # Add vertical line at T_train=5 cycles
    T_train = 5
    plt.axvline(x=T_train, color='gray', linestyle='--', linewidth=1.5, alpha=0.7,
                label=f'$T_{{\\text{{train}}}} = {T_train}$')
    
    # set x ticks according to the data and add labels only for every 5th tick
    plt.xticks(np.arange(0, max_cycles+1, 5)[1:])
    plt.ylim(0, max(energy_densities_train + energy_densities_test)*1.05)
    plt.xlabel('Cycle number', fontsize=FONT_SIZE_LABELS)
    plt.ylabel('Energy Density', fontsize=FONT_SIZE_LABELS)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=FONT_SIZE_LEGEND)
    plt.tick_params(labelsize=FONT_SIZE_TICKS)
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(FIGURES_DIR, 'energy_vs_cycles_24x24.pdf')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"  Saved to {fig_path}")
    plt.close()


def plot_energy_and_chern_vs_p(res_vals=None, p_vals=None, Jx=JX, Jy=JY, Jz=JZ, kappa=KAPPA):
    """
    Generate both energy_vs_p_by_res.pdf and chern_vs_p.pdf in a single pass.
    This avoids redundant computation by calling simulate_grid_with_analysis only once per grid.
    
    Args:
        res_vals: list of res values to plot (default: [1, 2, 3, 4])
        p_vals: list of p values to plot (default: [2, 3, 4, 5, 6, 7])
    """
    if res_vals is None:
        res_vals = [1, 2, 3, 4]
    if p_vals is None:
        p_vals = [2, 3, 4, 5, 6, 7]
    
    print(f"\n{'='*60}")
    print("Generating energy_vs_p_by_res.pdf and chern_vs_p.pdf (24×24 model)")
    print(f"{'='*60}")
    
    # Create test grid (fixed for all res values)
    n_k_points_test = 6 * 20  # 120
    kx_list_test = get_k_grid(n_k_points_test)
    ky_list_test = get_k_grid(n_k_points_test)
    
    # Initialize figure for energy plot
    plt.figure(figsize=FIG_SIZE)
    
    # Store data for both plots
    energy_data = {}  # {res: {'train': [...], 'test': [...], 'p_vals': [...]}}
    chern_data = {}   # {res: {'train': [...], 'test': [...], 'p_vals': [...]}}
    
    for res in res_vals:
        print(f"Processing res={res}...")
        
        # Create training grid for this res
        n_k_points_train = 6 * res
        kx_list_train = get_k_grid(n_k_points_train)
        ky_list_train = get_k_grid(n_k_points_train)
        
        # Initialize lists for this res
        energy_density_train_list = []
        energy_density_test_list = []
        system_chern_train_list = []
        system_chern_test_list = []
        p_vals_valid = []
        
        for p_val in p_vals:
            print(f"  Evaluating p={p_val}...")
            
            try:
                # Load parameters
                strength_durations = load_optimized_parameters_for_res_p(res, p_val, input_dir=PARAMS_DIR_24X24)
                
                # Run simulation on training grid (use N_CYCLES_STEADY for convergence)
                # Extract both energy_density and system_chern from single call
                _, _, _, system_chern_tr, _, energy_density_train = simulate_grid_with_analysis(
                    kx_list_train, ky_list_train, strength_durations, N_CYCLES_STEADY, Jx, Jy, Jz, kappa
                )
                
                # Run simulation on test grid (use N_CYCLES_STEADY for convergence)
                # Extract both energy_density and system_chern from single call
                _, _, _, system_chern_te, _, energy_density_test = simulate_grid_with_analysis(
                    kx_list_test, ky_list_test, strength_durations, N_CYCLES_STEADY, Jx, Jy, Jz, kappa
                )
                
                # Store all results
                energy_density_train_list.append(energy_density_train)
                energy_density_test_list.append(energy_density_test)
                system_chern_train_list.append(system_chern_tr)
                system_chern_test_list.append(system_chern_te)
                p_vals_valid.append(p_val)
                
                print(f"    Energy density (train): {energy_density_train:.6f}, Energy density (test): {energy_density_test:.6f}")
                print(f"    System $\\nu$ (train): {system_chern_tr:.4f}, System $\\nu$ (test): {system_chern_te:.4f}")
                
            except (FileNotFoundError, KeyError) as e:
                print(f"    Warning: Could not evaluate p={p_val}: {e}")
                continue
        
        if len(energy_density_test_list) == 0:
            print(f"  Warning: No valid data points for res={res}, skipping...")
            continue
        
        # Store data for this res
        energy_data[res] = {
            'train': energy_density_train_list,
            'test': energy_density_test_list,
            'p_vals': p_vals_valid
        }
        chern_data[res] = {
            'train': system_chern_train_list,
            'test': system_chern_test_list,
            'p_vals': p_vals_valid
        }
        
        # Plot energy data (test solid, train dashed)
        line_test, = plt.plot(p_vals_valid, energy_density_test_list, "-o", 
                             label=f"{res*6}", linewidth=2)
        color = line_test.get_color()
        plt.plot(p_vals_valid, energy_density_train_list, "--o", color=color, 
                alpha=0.8, linewidth=1.5)
    
    # Finalize energy plot
    plt.xlabel("$p$", fontsize=FONT_SIZE_LABELS)
    plt.ylabel("Energy density", fontsize=FONT_SIZE_LABELS)
    plt.grid(True, alpha=0.3)
    # set y to start from 0 and extend to 1.05 times the maximum value
    plt.ylim(0, max(energy_density_train_list + energy_density_test_list)*1.05)
    plt.legend(title="$L_{\\text{train}}$", title_fontsize=FONT_SIZE_LABELS, 
               fontsize=FONT_SIZE_LEGEND, ncol=2)
    plt.tick_params(labelsize=FONT_SIZE_TICKS)
    plt.tight_layout()
    
    # Save energy figure
    fig_path = os.path.join(FIGURES_DIR, 'energy_vs_p_by_res_24x24.pdf')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved energy_vs_p_by_res_24x24.pdf to {fig_path}")
    plt.close()
    
    # Create Chern plot
    plt.figure(figsize=FIG_SIZE)
    
    for res in res_vals:
        if res not in chern_data:
            continue
        
        data = chern_data[res]
        p_vals_valid = data['p_vals']
        system_chern_test = data['test']
        system_chern_train = data['train']
        
        # Plot test (solid) and train (dashed) with same color
        line_test, = plt.plot(p_vals_valid, system_chern_test, "-o", 
                             label=f"{res*6}", linewidth=2)
        color = line_test.get_color()
        plt.plot(p_vals_valid, system_chern_train, "--o", color=color, 
                alpha=0.8, linewidth=1.5)
    
    # Add horizontal line at target value
    plt.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    plt.xlabel("$p$", fontsize=FONT_SIZE_LABELS)
    plt.ylabel("$\\nu$", fontsize=FONT_SIZE_LABELS)
    plt.grid(True, alpha=0.3)
    plt.legend(title="$L_{\\text{train}}$", title_fontsize=FONT_SIZE_LABELS, 
               fontsize=FONT_SIZE_LEGEND, ncol=2)
    plt.tick_params(labelsize=FONT_SIZE_TICKS)
    plt.tight_layout()
    
    # Save Chern figure
    fig_path = os.path.join(FIGURES_DIR, 'chern_vs_p_24x24.pdf')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved chern_vs_p_24x24.pdf to {fig_path}")


def plot_energy_vs_p_by_res(res_vals=None, p_vals=None, Jx=JX, Jy=JY, Jz=JZ, kappa=KAPPA):
    """
    Generate energy_vs_p_by_res.pdf - Energy density vs. circuit depth p for different training system sizes.
    
    This function is a wrapper that calls the combined function for efficiency.
    For backward compatibility, it can be called independently, but it's more efficient
    to call plot_energy_and_chern_vs_p() if both plots are needed.
    
    Args:
        res_vals: list of res values to plot (default: [1, 2, 3, 4])
        p_vals: list of p values to plot (default: [2, 3, 4, 5, 6, 7])
    """
    # For backward compatibility, we'll still compute separately if called alone
    # But recommend using plot_energy_and_chern_vs_p for both
    if res_vals is None:
        res_vals = [1, 2, 3, 4]
    if p_vals is None:
        p_vals = [2, 3, 4, 5, 6, 7]
    
    print(f"\n{'='*60}")
    print("Generating energy_vs_p_by_res.pdf (24×24 model)")
    print(f"{'='*60}")
    
    # Create test grid (fixed for all res values)
    n_k_points_test = 6 * 20  # 120
    kx_list_test = get_k_grid(n_k_points_test)
    ky_list_test = get_k_grid(n_k_points_test)
    
    plt.figure(figsize=FIG_SIZE)
    
    for res in res_vals:
        print(f"Processing res={res}...")
        
        # Create training grid for this res
        n_k_points_train = 6 * res
        kx_list_train = get_k_grid(n_k_points_train)
        ky_list_train = get_k_grid(n_k_points_train)
        
        # Calculate energy densities for training and test grids
        energy_density_train_list = []
        energy_density_test_list = []
        p_vals_valid = []
        
        for p_val in p_vals:
            print(f"  Evaluating p={p_val}...")
            
            try:
                # Load parameters
                strength_durations = load_optimized_parameters_for_res_p(res, p_val, input_dir=PARAMS_DIR_24X24)
                
                # Run simulation on training grid (use N_CYCLES_STEADY for convergence)
                _, _, _, _, _, energy_density_train = simulate_grid_with_analysis(
                    kx_list_train, ky_list_train, strength_durations, N_CYCLES_STEADY, Jx, Jy, Jz, kappa
                )
                
                # Run simulation on test grid (use N_CYCLES_STEADY for convergence)
                _, _, _, _, _, energy_density_test = simulate_grid_with_analysis(
                    kx_list_test, ky_list_test, strength_durations, N_CYCLES_STEADY, Jx, Jy, Jz, kappa
                )
                
                energy_density_train_list.append(energy_density_train)
                energy_density_test_list.append(energy_density_test)
                p_vals_valid.append(p_val)
                
                print(f"    Energy density (train): {energy_density_train:.6f}, Energy density (test): {energy_density_test:.6f}")
                
            except (FileNotFoundError, KeyError) as e:
                print(f"    Warning: Could not evaluate p={p_val}: {e}")
                continue
        
        if len(energy_density_test_list) == 0:
            print(f"  Warning: No valid data points for res={res}, skipping...")
            continue
        
        # Plot test (solid) and train (dashed) with same color
        line_test, = plt.plot(p_vals_valid, energy_density_test_list, "-o", 
                             label=f"{res*6}", linewidth=2)
        color = line_test.get_color()
        plt.plot(p_vals_valid, energy_density_train_list, "--o", color=color, 
                alpha=0.8, linewidth=1.5)
    
    plt.xlabel("$p$", fontsize=FONT_SIZE_LABELS)
    plt.ylabel("Energy density", fontsize=FONT_SIZE_LABELS)
    plt.grid(True, alpha=0.3)
    plt.legend(title="$L_{\\text{train}}$", title_fontsize=FONT_SIZE_LABELS, 
               fontsize=FONT_SIZE_LEGEND, ncol=2)
    plt.tick_params(labelsize=FONT_SIZE_TICKS)
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(FIGURES_DIR, 'energy_vs_p_by_res_24x24.pdf')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved energy_vs_p_by_res_24x24.pdf to {fig_path}")
    plt.close()


def plot_chern_vs_p(res_vals=None, p_vals=None, Jx=JX, Jy=JY, Jz=JZ, kappa=KAPPA):
    """
    Generate chern_vs_p.pdf - Finite size spectral Chern number (ν) vs. circuit depth p.
    
    This function is a wrapper that calls the combined function for efficiency.
    For backward compatibility, it can be called independently, but it's more efficient
    to call plot_energy_and_chern_vs_p() if both plots are needed.
    
    Args:
        res_vals: list of res values to plot (default: [1, 2, 3, 4])
        p_vals: list of p values to plot (default: [2, 3, 4, 5, 6, 7])
    """
    # For backward compatibility, we'll still compute separately if called alone
    # But recommend using plot_energy_and_chern_vs_p for both
    if res_vals is None:
        res_vals = [1, 2, 3, 4]
    if p_vals is None:
        p_vals = [2, 3, 4, 5, 6, 7]
    
    print(f"\n{'='*60}")
    print("Generating chern_vs_p.pdf (24×24 model)")
    print(f"{'='*60}")
    
    # Create test grid (fixed for all res values)
    n_k_points_test = 6 * 20  # 120
    kx_list_test = get_k_grid(n_k_points_test)
    ky_list_test = get_k_grid(n_k_points_test)
    
    plt.figure(figsize=FIG_SIZE)
    
    for res in res_vals:
        print(f"Processing res={res}...")
        
        # Create training grid for this res
        n_k_points_train = 6 * res
        kx_list_train = get_k_grid(n_k_points_train)
        ky_list_train = get_k_grid(n_k_points_train)
        
        # Calculate Chern numbers for training and test grids
        system_chern_train = []
        system_chern_test = []
        p_vals_valid = []
        
        for p_val in p_vals:
            print(f"  Evaluating p={p_val}...")
            
            try:
                # Load parameters
                strength_durations = load_optimized_parameters_for_res_p(res, p_val, input_dir=PARAMS_DIR_24X24)
                
                # Run simulation on training grid (use N_CYCLES_STEADY for convergence)
                _, _, _, system_chern_tr, _, _ = simulate_grid_with_analysis(
                    kx_list_train, ky_list_train, strength_durations, N_CYCLES_STEADY, Jx, Jy, Jz, kappa
                )
                
                # Run simulation on test grid (use N_CYCLES_STEADY for convergence)
                _, _, _, system_chern_te, _, _ = simulate_grid_with_analysis(
                    kx_list_test, ky_list_test, strength_durations, N_CYCLES_STEADY, Jx, Jy, Jz, kappa
                )
                
                system_chern_train.append(system_chern_tr)
                system_chern_test.append(system_chern_te)
                p_vals_valid.append(p_val)
                
                print(f"    System $\\nu$ (train): {system_chern_tr:.4f}, System $\\nu$ (test): {system_chern_te:.4f}")
                
            except (FileNotFoundError, KeyError) as e:
                print(f"    Warning: Could not evaluate p={p_val}: {e}")
                continue
        
        if len(system_chern_test) == 0:
            print(f"  Warning: No valid data points for res={res}, skipping...")
            continue
        
        # Plot test (solid) and train (dashed) with same color
        line_test, = plt.plot(p_vals_valid, system_chern_test, "-o", 
                             label=f"{res*6}", linewidth=2)
        color = line_test.get_color()
        plt.plot(p_vals_valid, system_chern_train, "--o", color=color, 
                alpha=0.8, linewidth=1.5)
    
    # Add horizontal line at target value
    plt.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    plt.xlabel("$p$", fontsize=FONT_SIZE_LABELS)
    plt.ylabel("$\\nu$", fontsize=FONT_SIZE_LABELS)
    plt.grid(True, alpha=0.3)
    plt.legend(title="$L_{\\text{train}}$", title_fontsize=FONT_SIZE_LABELS, 
               fontsize=FONT_SIZE_LEGEND, ncol=2)
    plt.tick_params(labelsize=FONT_SIZE_TICKS)
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(FIGURES_DIR, 'chern_vs_p_24x24.pdf')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved chern_vs_p_24x24.pdf to {fig_path}")


def plot_energy_heatmap(res_val=3, p_val=5, grid_type='train', Jx=JX, Jy=JY, Jz=JZ, kappa=KAPPA):
    """
    Generate energy heatmap - Energy difference heatmap on training or test grid
    
    Args:
        res_val: resolution parameter
        p_val: number of layers
        grid_type: 'train' or 'test' - determines which grid to use
        Jx, Jy, Jz, kappa: Kitaev parameters
    """
    grid_name = 'train' if grid_type == 'train' else 'test'
    print(f"\n{'='*60}")
    print(f"Generating energy_heatmap_{grid_name}.pdf (24×24 model)")
    print(f"{'='*60}")
    
    # Load parameters
    strength_durations = load_optimized_parameters_for_res_p(res_val, p_val, input_dir=PARAMS_DIR_24X24)
    
    # Determine grid size based on grid type
    if grid_type == 'train':
        n_k_points = 6 * res_val
        show_training_points = False
    else:  # test
        n_k_points = 6 * 20  # 120
        show_training_points = False
        n_k_points_train = 6 * res_val
    
    # Create grid
    kx_list = get_k_grid(n_k_points)
    ky_list = get_k_grid(n_k_points)
    
    # Run simulation (use N_CYCLES_STEADY for convergence)
    print(f"  Running simulation on {grid_name} grid ({n_k_points}x{n_k_points}) with {N_CYCLES_STEADY} cycles...")
    E_diff, _, _, _, _, _ = simulate_grid_with_analysis(
        kx_list, ky_list, strength_durations, N_CYCLES_STEADY, Jx, Jy, Jz, kappa
    )
    
    # Extract final cycle if multiple cycles
    if E_diff.ndim == 3:
        E_diff_plot = E_diff[-1, :, :]
    else:
        E_diff_plot = E_diff
    
    # Create plot with adjusted layout for larger grid
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    # Adjust subplot to give more space to the main plot
    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.95)
    
    kx_coords = get_k_grid(n_k_points)
    ky_coords = get_k_grid(n_k_points)
    
    vmin = 0.0
    vmax = np.max(E_diff_plot)
    im = ax.pcolormesh(kx_coords, ky_coords, E_diff_plot, vmin=vmin, vmax=vmax, shading='auto')
    
    # Create colorbar with reduced padding to bring it closer
    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.9)
    cbar.set_label('Energy difference', fontsize=FONT_SIZE_LABELS)
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICKS)
    
    # Mark training points if this is test grid
    if show_training_points:
        step = (n_k_points - 1) / (n_k_points_train - 1)
        training_indices = np.round(np.arange(0, n_k_points, step)).astype(int)
        training_indices = training_indices[:n_k_points_train]
        
        kx_train_coords = kx_coords[training_indices]
        ky_train_coords = ky_coords[training_indices]
        
        for i, kx_coord in enumerate(kx_train_coords):
            for j, ky_coord in enumerate(ky_train_coords):
                ax.scatter(kx_coord, ky_coord, c='red', marker='x', s=50, linewidths=2, 
                           label='Training points' if i==0 and j==0 else "")
    
    ax.set_xlabel('$k_x$', fontsize=FONT_SIZE_LABELS)
    ax.set_ylabel('$k_y$', fontsize=FONT_SIZE_LABELS)
    if show_training_points:
        ax.legend(fontsize=FONT_SIZE_LEGEND)
    ax.tick_params(labelsize=FONT_SIZE_TICKS)
    ax.set_aspect('equal')
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    # set the ticks to be -pi, 0, pi
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_yticks([-np.pi, 0, np.pi])
    ax.set_xticklabels(['$-\\pi$', '$0$', '$\\pi$'])
    ax.set_yticklabels(['$-\\pi$', '$0$', '$\\pi$'])
    
    # Save figure
    fig_path = os.path.join(FIGURES_DIR, f'energy_heatmap_{grid_name}_24x24.pdf')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"  Saved to {fig_path}")
    plt.close()


def plot_energy_heatmap_train(res_val=3, p_val=5, Jx=JX, Jy=JY, Jz=JZ, kappa=KAPPA):
    """Wrapper for backward compatibility"""
    plot_energy_heatmap(res_val, p_val, grid_type='train', Jx=Jx, Jy=Jy, Jz=Jz, kappa=kappa)


def plot_energy_heatmap_test(res_val=3, p_val=5, Jx=JX, Jy=JY, Jz=JZ, kappa=KAPPA):
    """Wrapper for backward compatibility"""
    plot_energy_heatmap(res_val, p_val, grid_type='test', Jx=Jx, Jy=Jy, Jz=Jz, kappa=kappa)


def plot_gap_heatmap(res_val=3, p_val=5, grid_type='train', Jx=JX, Jy=JY, Jz=JZ, kappa=KAPPA):
    """
    Generate spectral gap heatmap - Smallest positive eigenvalue of the Hamiltonian at each k-point.
    
    The spectral gap is the smallest positive eigenvalue of the system Hamiltonian (8x8 block
    for c^z fermions) at each k-point. This determines the energy scale for excitations.
    
    Args:
        res_val: resolution parameter (only used for naming, not for computation)
        p_val: number of layers (only used for naming, not for computation)
        grid_type: 'train' or 'test' - determines which grid to use
        Jx, Jy, Jz, kappa: Kitaev parameters
    """
    grid_name = 'train' if grid_type == 'train' else 'test'
    print(f"\n{'='*60}")
    print(f"Generating gap_heatmap_{grid_name}.pdf (24×24 model)")
    print(f"{'='*60}")
    
    # Determine grid size based on grid type
    if grid_type == 'train':
        n_k_points = 6 * res_val
    else:  # test
        n_k_points = 6 * 20  # 120
    
    # Create grid
    kx_list = get_k_grid(n_k_points)
    ky_list = get_k_grid(n_k_points)
    
    # Compute spectral gap at each k-point
    print(f"  Computing spectral gap on {grid_name} grid ({n_k_points}x{n_k_points})...")
    gap_at_k = np.zeros((len(kx_list), len(ky_list)), dtype=float)
    
    for i_kx, kx in enumerate(kx_list):
        for i_ky, ky in enumerate(ky_list):
            # Create system Hamiltonian (g=0, B=0 for ground state)
            hamiltonian = create_KSL_24x24_hamiltonian(
                kx, ky, Jx=Jx, Jy=Jy, Jz=Jz, kappa=kappa, g=0.0, B=0.0
            )
            
            # Get the 8x8 system block (c^z fermions, modes 0-7)
            H_full = hamiltonian.get_matrix()
            H_system = H_full[:8, :8]
            
            # Compute eigenvalues
            eigenvalues = np.linalg.eigvalsh(H_system)
            
            # Find smallest positive eigenvalue (spectral gap)
            positive_eigenvalues = eigenvalues[eigenvalues > 1e-10]
            if len(positive_eigenvalues) > 0:
                gap_at_k[i_kx, i_ky] = np.min(positive_eigenvalues)
            else:
                gap_at_k[i_kx, i_ky] = 0.0
    
    print(f"  Min spectral gap: {gap_at_k.min():.4f}, Max spectral gap: {gap_at_k.max():.4f}")
    
    # Create plot with adjusted layout
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    # Adjust subplot to give more space to the main plot
    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.95)
    
    kx_coords = get_k_grid(n_k_points)
    ky_coords = get_k_grid(n_k_points)
    
    vmin = 0.0
    vmax = np.max(gap_at_k)
    im = ax.pcolormesh(kx_coords, ky_coords, gap_at_k.T, vmin=vmin, vmax=vmax, shading='auto', cmap='viridis')
    
    # Create colorbar with reduced padding to bring it closer
    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.9)
    cbar.set_label('Spectral gap', fontsize=FONT_SIZE_LABELS)
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICKS)
    
    ax.set_xlabel('$k_x$', fontsize=FONT_SIZE_LABELS)
    ax.set_ylabel('$k_y$', fontsize=FONT_SIZE_LABELS)
    ax.tick_params(labelsize=FONT_SIZE_TICKS)
    ax.set_aspect('equal')
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    # set the ticks to be -pi, 0, pi
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_yticks([-np.pi, 0, np.pi])
    ax.set_xticklabels(['$-\\pi$', '$0$', '$\\pi$'])
    ax.set_yticklabels(['$-\\pi$', '$0$', '$\\pi$'])
    
    # Save figure
    fig_path = os.path.join(FIGURES_DIR, f'gap_heatmap_{grid_name}_24x24.pdf')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"  Saved to {fig_path}")
    plt.close()


def plot_gap_heatmap_train(res_val=3, p_val=5, Jx=JX, Jy=JY, Jz=JZ, kappa=KAPPA):
    """Wrapper for train grid gap heatmap"""
    plot_gap_heatmap(res_val, p_val, grid_type='train', Jx=Jx, Jy=Jy, Jz=Jz, kappa=kappa)


def plot_gap_heatmap_test(res_val=3, p_val=5, Jx=JX, Jy=JY, Jz=JZ, kappa=KAPPA):
    """Wrapper for test grid gap heatmap"""
    plot_gap_heatmap(res_val, p_val, grid_type='test', Jx=Jx, Jy=Jy, Jz=Jz, kappa=kappa)


def plot_chern_vs_test_grid_size(res_val=2, p_val=5, test_grid_sizes=None, Jx=JX, Jy=JY, Jz=JZ, kappa=KAPPA):
    """
    Generate chern_vs_test_grid_size.pdf - Chern number vs. test grid size
    
    Args:
        res_val: resolution parameter (training grid size = 6 * res_val)
        p_val: number of layers
        test_grid_sizes: list of test grid sizes to evaluate (default: [6, 12, 18, 24, 30, 36, 42, 48, 60, 90, 120])
        Jx, Jy, Jz, kappa: Kitaev parameters
    """
    if test_grid_sizes is None:
        test_grid_sizes = [6, 12, 24, 48, 120, 240]
    
    print(f"\n{'='*60}")
    print("Generating chern_vs_test_grid_size.pdf (24×24 model)")
    print(f"{'='*60}")
    print(f"Training grid size: {6 * res_val}")
    print(f"Test grid sizes: {test_grid_sizes}")
    
    # Load parameters
    strength_durations = load_optimized_parameters_for_res_p(res_val, p_val, input_dir=PARAMS_DIR_24X24)
    
    # Calculate Chern numbers for different test grid sizes
    system_chern_list = []
    test_grid_sizes_valid = []
    
    for n_k_points_test in test_grid_sizes:
        print(f"  Evaluating test grid size: {n_k_points_test}...")
        
        try:
            # Create test grid
            kx_list_test = get_k_grid(n_k_points_test)
            ky_list_test = get_k_grid(n_k_points_test)
            
            # Run simulation on test grid (use N_CYCLES_STEADY for convergence)
            _, _, _, system_chern, _, _ = simulate_grid_with_analysis(
                kx_list_test, ky_list_test, strength_durations, N_CYCLES_STEADY, Jx, Jy, Jz, kappa
            )
            
            system_chern_list.append(system_chern)
            test_grid_sizes_valid.append(n_k_points_test)
            
            print(f"    System $\\nu$: {system_chern:.6f}")
            
        except Exception as e:
            print(f"    Warning: Could not evaluate grid size {n_k_points_test}: {e}")
            continue
    
    if len(system_chern_list) == 0:
        print("  Error: No valid data points, skipping plot...")
        return
    
    # Create plot
    plt.figure(figsize=FIG_SIZE)
    plt.plot(test_grid_sizes_valid, system_chern_list, 'b-o', linewidth=2, markersize=6)
    
    # Add horizontal line at target value
    plt.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, linewidth=1, label='Target $\\nu=1$')
    
    plt.xlabel('Test grid size $L_{\\text{test}}$', fontsize=FONT_SIZE_LABELS)
    plt.ylabel('$\\nu$', fontsize=FONT_SIZE_LABELS)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=FONT_SIZE_LEGEND)
    plt.tick_params(labelsize=FONT_SIZE_TICKS)
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(FIGURES_DIR, 'chern_vs_test_grid_size_24x24.pdf')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"  Saved to {fig_path}")
    plt.close()


def plot_parameters_vs_layer(res_val=3, p_val=5):
    """
    Generate parameters_vs_layer.pdf - Optimized parameters vs. layer number
    
    Args:
        res_val: resolution parameter
        p_val: number of layers
    """
    print(f"\n{'='*60}")
    print("Generating parameters_vs_layer.pdf (24×24 model)")
    print(f"{'='*60}")
    
    # Load parameters
    strength_durations = load_optimized_parameters_for_res_p(res_val, p_val, input_dir=PARAMS_DIR_24X24)
    
    # Create plot
    plt.figure(figsize=FIG_SIZE)
    
    terms = ['Jx', 'Jy', 'Jz', 'kappa', 'g', 'B']
    labels = ['$\\alpha_{x,\\ell}$ ($J_x$)', '$\\alpha_{y,\\ell}$ ($J_y$)', '$\\alpha_{z,\\ell}$ ($J_z$)', 
              '$\\delta_\\ell$ ($\\kappa$)', '$\\gamma_\\ell$ ($g$)', '$\\beta_\\ell$ ($B$)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    linestyles = ['-', '-', '-', '-', '-', '-']
    
    for i, (term, label, color, ls) in enumerate(zip(terms, labels, colors, linestyles)):
        # Wrap parameters to [-π, π] since they are periodic mod 2π
        params_wrapped = np.mod(strength_durations[term] + np.pi, 2 * np.pi) - np.pi
        print(f"params_wrapped for {term}: {params_wrapped}")
        plt.plot(range(1, p_val + 1), params_wrapped, 
                label=label, marker='o', linewidth=2, markersize=6, color=color, linestyle=ls)
    
    # set x ticks according to the data
    plt.xticks(np.arange(1, p_val + 1))
    plt.yticks([-np.pi, 0, np.pi], ['$-\\pi$', '$0$', '$\\pi$'])
    plt.xlabel('Layer number $\\ell$', fontsize=FONT_SIZE_LABELS)
    plt.ylabel('Parameter value', fontsize=FONT_SIZE_LABELS)
    plt.legend(fontsize=FONT_SIZE_LEGEND, ncol=2)
    plt.tick_params(labelsize=FONT_SIZE_TICKS)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(FIGURES_DIR, 'parameters_vs_layer_24x24.pdf')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"  Saved to {fig_path}")
    plt.close()


def main(res_val=2, p_val=5, res_vals=None, p_vals=None, generate_all=True, 
         Jx=JX, Jy=JY, Jz=JZ, kappa=KAPPA):
    """
    Generate all figures for the LaTeX document (24×24 model)
    
    Args:
        res_val: resolution parameter for single-res plots (default: 3)
        p_val: number of layers for single-p plots (default: 5)
        res_vals: list of res values for multi-res plots (default: [1, 2, 3, 4])
        p_vals: list of p values for multi-p plots (default: [2, 3, 4, 5, 6, 7])
        generate_all: if True, generate all figures; if False, only generate single-res plots
        Jx, Jy, Jz, kappa: Kitaev parameters
    """
    print("="*60)
    print("GENERATING ALL FIGURES FOR LATEX SUMMARY (24×24 MODEL)")
    print("="*60)
    print(f"Using {N_CYCLES_STEADY} cycles for convergence to steady state")
    print(f"Using res={res_val}, p={p_val} for single-res plots")
    print(f"Kitaev parameters: Jx={Jx}, Jy={Jy}, Jz={Jz}, kappa={kappa}")
    print(f"Parameters directory: {PARAMS_DIR_24X24}")
    print(f"Figures will be saved to: {FIGURES_DIR}")
    print("="*60)
    
    # Generate single-res figures
    # plot_chern_vs_test_grid_size(res_val, p_val, Jx=Jx, Jy=Jy, Jz=Jz, kappa=kappa)
    plot_parameters_vs_layer(res_val, p_val)
    plot_energy_vs_cycles(res_val, p_val, max_cycles=20, Jx=Jx, Jy=Jy, Jz=Jz, kappa=kappa)
    plot_energy_heatmap_train(res_val, p_val, Jx=Jx, Jy=Jy, Jz=Jz, kappa=kappa)
    plot_energy_heatmap_test(res_val, p_val, Jx=Jx, Jy=Jy, Jz=Jz, kappa=kappa)
    plot_gap_heatmap_train(res_val, p_val, Jx=Jx, Jy=Jy, Jz=Jz, kappa=kappa)
    plot_gap_heatmap_test(res_val, p_val, Jx=Jx, Jy=Jy, Jz=Jz, kappa=kappa)
    
    # Generate multi-res figures (if requested)
    # Use combined function to avoid redundant computation
    if generate_all:
        plot_energy_and_chern_vs_p(res_vals=res_vals, p_vals=p_vals, Jx=Jx, Jy=Jy, Jz=Jz, kappa=kappa)
    
    print("\n" + "="*60)
    print("ALL FIGURES GENERATED SUCCESSFULLY (24×24 MODEL)")
    print("="*60)
    print(f"Figures saved to: {FIGURES_DIR}")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate all figures for LaTeX summary (24×24 model)')
    parser.add_argument('--res', type=int, default=2, help='Resolution parameter for single-res plots (default: 2)')
    parser.add_argument('--p', type=int, default=5, help='Number of layers for single-p plots (default: 5)')
    parser.add_argument('--res-vals', type=int, nargs='+', default=[1, 2, 3, 4], 
                       help='List of res values for multi-res plots (default: [1, 2, 3, 4])')
    parser.add_argument('--p-vals', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7],
                       help='List of p values for multi-p plots (default: [2, 3, 4, 5, 6, 7])')
    parser.add_argument('--single-only', action='store_true',
                       help='Only generate single-res plots (skip energy_vs_p_by_res and chern_vs_p)')
    parser.add_argument('--Jx', type=float, default=JX, help=f'Jx parameter (default: {JX})')
    parser.add_argument('--Jy', type=float, default=JY, help=f'Jy parameter (default: {JY})')
    parser.add_argument('--Jz', type=float, default=JZ, help=f'Jz parameter (default: {JZ})')
    parser.add_argument('--kappa', type=float, default=KAPPA, help=f'kappa parameter (default: {KAPPA})')
    
    args = parser.parse_args()
    
    main(res_val=args.res, p_val=args.p, res_vals=args.res_vals, p_vals=args.p_vals, 
         generate_all=not args.single_only, Jx=args.Jx, Jy=args.Jy, Jz=args.Jz, kappa=args.kappa)

