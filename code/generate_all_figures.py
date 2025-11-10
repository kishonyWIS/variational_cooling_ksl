"""
Generate all figures for the LaTeX summary document.

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

# Add parent directory to path to import from variational_circuit_KSL_numba
sys.path.insert(0, os.path.dirname(__file__))

from variational_circuit_KSL_numba import (
    load_optimized_parameters_for_res_p,
    simulate_grid_with_analysis,
    simulate_grid,
    DATA_DIR,
)

# Figure output directory
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '../figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Number of cycles for convergence to steady state (used for all plots)
N_CYCLES_STEADY = 40

# Consistent formatting constants
FIG_SIZE = (7, 4)  # Standard figure size for all plots

FONT_SIZE_LABELS = 20
FONT_SIZE_LEGEND = 17
FONT_SIZE_TICKS = 17


def plot_energy_vs_cycles(res_val=3, p_val=5, max_cycles=20):
    """
    Generate energy_vs_cycles.pdf showing energy density vs. number of cooling cycles
    
    Args:
        res_val: resolution parameter
        p_val: number of layers
        max_cycles: maximum number of cycles to plot
    """
    print(f"\n{'='*60}")
    print("Generating energy_vs_cycles.pdf")
    print(f"{'='*60}")
    
    # Load parameters
    strength_durations = load_optimized_parameters_for_res_p(res_val, p_val)
    
    # Set global p (needed for create_variational_circuit)
    import variational_circuit_KSL_numba as vc_module
    vc_module.p = p_val
    
    # Create grids
    n_k_points_train = 6 * res_val
    n_k_points_test = 6 * 20  # 120
    
    kx_list_train = np.linspace(-np.pi, np.pi, n_k_points_train + 1)[:-1]
    ky_list_train = np.linspace(-np.pi, np.pi, n_k_points_train + 1)[:-1]
    kx_list_test = np.linspace(-np.pi, np.pi, n_k_points_test + 1)[:-1]
    ky_list_test = np.linspace(-np.pi, np.pi, n_k_points_test + 1)[:-1]
    
    # Run simulation for training grid
    print(f"  Running simulation on training grid ({n_k_points_train}x{n_k_points_train}) with {max_cycles} cycles...")
    E_diff_train_all_cycles, _ = simulate_grid(kx_list_train, ky_list_train, strength_durations, max_cycles, verbose=False)
    
    # Run simulation for test grid
    print(f"  Running simulation on test grid ({n_k_points_test}x{n_k_points_test}) with {max_cycles} cycles...")
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
    fig_path = os.path.join(FIGURES_DIR, 'energy_vs_cycles.pdf')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"  Saved to {fig_path}")
    plt.close()


def plot_energy_vs_p_by_res(res_vals=None, p_vals=None):
    """
    Generate energy_vs_p_by_res.pdf - Energy density vs. circuit depth p for different training system sizes
    
    Args:
        res_vals: list of res values to plot (default: [1, 2, 3, 4])
        p_vals: list of p values to plot (default: [2, 3, 4, 5, 6, 7, 8, 9, 10])
    """
    if res_vals is None:
        res_vals = [1, 2, 3, 4]
    if p_vals is None:
        p_vals = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    print(f"\n{'='*60}")
    print("Generating energy_vs_p_by_res.pdf")
    print(f"{'='*60}")
    
    import variational_circuit_KSL_numba as vc_module
    
    # Create test grid (fixed for all res values)
    n_k_points_test = 6 * 20  # 120
    kx_list_test = np.linspace(-np.pi, np.pi, n_k_points_test + 1)[:-1]
    ky_list_test = np.linspace(-np.pi, np.pi, n_k_points_test + 1)[:-1]
    
    plt.figure(figsize=FIG_SIZE)
    
    for res in res_vals:
        print(f"Processing res={res}...")
        
        # Create training grid for this res
        n_k_points_train = 6 * res
        kx_list_train = np.linspace(-np.pi, np.pi, n_k_points_train + 1)[:-1]
        ky_list_train = np.linspace(-np.pi, np.pi, n_k_points_train + 1)[:-1]
        
        # Calculate energy densities for training and test grids
        energy_density_train_list = []
        energy_density_test_list = []
        p_vals_valid = []
        
        for p_val in p_vals:
            print(f"  Evaluating p={p_val}...")
            
            try:
                # Load parameters
                strength_durations = load_optimized_parameters_for_res_p(res, p_val)
                
                # Set global p
                vc_module.p = p_val
                
                # Run simulation on training grid (use N_CYCLES_STEADY for convergence)
                _, _, _, _, _, energy_density_train = simulate_grid_with_analysis(
                    kx_list_train, ky_list_train, strength_durations, n_cycles=N_CYCLES_STEADY
                )
                
                # Run simulation on test grid (use N_CYCLES_STEADY for convergence)
                _, _, _, _, _, energy_density_test = simulate_grid_with_analysis(
                    kx_list_test, ky_list_test, strength_durations, n_cycles=N_CYCLES_STEADY
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
    fig_path = os.path.join(FIGURES_DIR, 'energy_vs_p_by_res.pdf')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved energy_vs_p_by_res.pdf to {fig_path}")
    plt.close()


def plot_chern_vs_p(res_vals=None, p_vals=None):
    """
    Generate chern_vs_p.pdf - Finite size spectral Chern number (ν) vs. circuit depth p
    
    Args:
        res_vals: list of res values to plot (default: [1, 2, 3, 4])
        p_vals: list of p values to plot (default: [2, 3, 4, 5, 6, 7, 8, 9, 10])
    """
    if res_vals is None:
        res_vals = [1, 2, 3, 4]
    if p_vals is None:
        p_vals = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    print(f"\n{'='*60}")
    print("Generating chern_vs_p.pdf")
    print(f"{'='*60}")
    
    import variational_circuit_KSL_numba as vc_module
    
    # Create test grid (fixed for all res values)
    # include the endpoint because of finite derivative for chern number
    n_k_points_test = 6 * 20  # 120
    kx_list_test = np.linspace(-np.pi, np.pi, n_k_points_test + 1)
    ky_list_test = np.linspace(-np.pi, np.pi, n_k_points_test + 1)
    
    plt.figure(figsize=FIG_SIZE)
    
    for res in res_vals:
        print(f"Processing res={res}...")
        
        # Create training grid for this res
        # include the endpoint because of finite derivative for chern number
        n_k_points_train = 6 * res
        kx_list_train = np.linspace(-np.pi, np.pi, n_k_points_train + 1)
        ky_list_train = np.linspace(-np.pi, np.pi, n_k_points_train + 1)
        
        # Calculate Chern numbers for training and test grids
        system_chern_train = []
        system_chern_test = []
        p_vals_valid = []
        
        for p_val in p_vals:
            print(f"  Evaluating p={p_val}...")
            
            try:
                # Load parameters
                strength_durations = load_optimized_parameters_for_res_p(res, p_val)
                
                # Set global p
                vc_module.p = p_val
                
                # Run simulation on training grid (use N_CYCLES_STEADY for convergence)
                _, _, _, system_chern_tr, _, _ = simulate_grid_with_analysis(
                    kx_list_train, ky_list_train, strength_durations, n_cycles=N_CYCLES_STEADY
                )
                
                # Run simulation on test grid (use N_CYCLES_STEADY for convergence)
                _, _, _, system_chern_te, _, _ = simulate_grid_with_analysis(
                    kx_list_test, ky_list_test, strength_durations, n_cycles=N_CYCLES_STEADY
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
    fig_path = os.path.join(FIGURES_DIR, 'chern_vs_p.pdf')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved chern_vs_p.pdf to {fig_path}")


def plot_energy_heatmap(res_val=3, p_val=5, grid_type='train'):
    """
    Generate energy heatmap - Energy difference heatmap on training or test grid
    
    Args:
        res_val: resolution parameter
        p_val: number of layers
        grid_type: 'train' or 'test' - determines which grid to use
    """
    grid_name = 'train' if grid_type == 'train' else 'test'
    print(f"\n{'='*60}")
    print(f"Generating energy_heatmap_{grid_name}.pdf")
    print(f"{'='*60}")
    
    # Load parameters
    strength_durations = load_optimized_parameters_for_res_p(res_val, p_val)
    
    # Set global p
    import variational_circuit_KSL_numba as vc_module
    vc_module.p = p_val
    
    # Determine grid size based on grid type
    if grid_type == 'train':
        n_k_points = 6 * res_val
        show_training_points = False
    else:  # test
        n_k_points = 6 * 20  # 120
        show_training_points = True
        n_k_points_train = 6 * res_val
    
    # Create grid
    kx_list = np.linspace(-np.pi, np.pi, n_k_points + 1)[:-1]
    ky_list = np.linspace(-np.pi, np.pi, n_k_points + 1)[:-1]
    
    # Run simulation (use N_CYCLES_STEADY for convergence)
    print(f"  Running simulation on {grid_name} grid ({n_k_points}x{n_k_points}) with {N_CYCLES_STEADY} cycles...")
    E_diff, _, _, _, _, _ = simulate_grid_with_analysis(
        kx_list, ky_list, strength_durations, n_cycles=N_CYCLES_STEADY
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
    
    kx_coords = np.linspace(-np.pi, np.pi, n_k_points + 1)[:-1]
    ky_coords = np.linspace(-np.pi, np.pi, n_k_points + 1)[:-1]
    
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
    
    # Save figure
    fig_path = os.path.join(FIGURES_DIR, f'energy_heatmap_{grid_name}.pdf')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"  Saved to {fig_path}")
    plt.close()


def plot_energy_heatmap_train(res_val=3, p_val=5):
    """Wrapper for backward compatibility"""
    plot_energy_heatmap(res_val, p_val, grid_type='train')


def plot_energy_heatmap_test(res_val=3, p_val=5):
    """Wrapper for backward compatibility"""
    plot_energy_heatmap(res_val, p_val, grid_type='test')


def plot_parameters_vs_layer(res_val=3, p_val=5):
    """
    Generate parameters_vs_layer.pdf - Optimized parameters vs. layer number
    
    Args:
        res_val: resolution parameter
        p_val: number of layers
    """
    print(f"\n{'='*60}")
    print("Generating parameters_vs_layer.pdf")
    print(f"{'='*60}")
    
    # Load parameters
    strength_durations = load_optimized_parameters_for_res_p(res_val, p_val)
    
    # Create plot
    plt.figure(figsize=FIG_SIZE)
    
    terms = ['Jx', 'Jy', 'Jz', 'kappa', 'g', 'B']
    labels = ['$\\alpha_{x,\\ell}$ (Jx)', '$\\alpha_{y,\\ell}$ (Jy)', '$\\alpha_{z,\\ell}$ (Jz)', 
              '$\\delta_\\ell$ ($\\kappa$)', '$\\gamma_\\ell$ (g)', '$\\beta_\\ell$ (B)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    linestyles = ['-', '-', '-', '-', '-', '-']
    
    for i, (term, label, color, ls) in enumerate(zip(terms, labels, colors, linestyles)):
        plt.plot(range(1, p_val + 1), strength_durations[term], 
                label=label, marker='o', linewidth=2, markersize=6, color=color, linestyle=ls)
    
    # set x ticks according to the data
    plt.xticks(np.arange(1, p_val + 1))
    plt.xlabel('Layer number $\\ell$', fontsize=FONT_SIZE_LABELS)
    plt.ylabel('Parameter value', fontsize=FONT_SIZE_LABELS)
    plt.legend(fontsize=FONT_SIZE_LEGEND, ncol=2)
    plt.tick_params(labelsize=FONT_SIZE_TICKS)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(FIGURES_DIR, 'parameters_vs_layer.pdf')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"  Saved to {fig_path}")
    plt.close()


def main(res_val=3, p_val=5, res_vals=None, p_vals=None, generate_all=True):
    """
    Generate all figures for the LaTeX document
    
    Args:
        res_val: resolution parameter for single-res plots (default: 3)
        p_val: number of layers for single-p plots (default: 5)
        res_vals: list of res values for multi-res plots (default: [1, 2, 3, 4])
        p_vals: list of p values for multi-p plots (default: [2, 3, 4, 5, 6, 7, 8, 9, 10])
        generate_all: if True, generate all figures; if False, only generate single-res plots
    """
    print("="*60)
    print("GENERATING ALL FIGURES FOR LATEX SUMMARY")
    print("="*60)
    print(f"Using {N_CYCLES_STEADY} cycles for convergence to steady state")
    print(f"Using res={res_val}, p={p_val} for single-res plots")
    print(f"Figures will be saved to: {FIGURES_DIR}")
    print("="*60)
    
    # Generate single-res figures
    plot_energy_vs_cycles(res_val, p_val, max_cycles=20)
    plot_energy_heatmap_train(res_val, p_val)
    plot_energy_heatmap_test(res_val, p_val)
    plot_parameters_vs_layer(res_val, p_val)
    
    # Generate multi-res figures (if requested)
    if generate_all:
        plot_energy_vs_p_by_res(res_vals=res_vals, p_vals=p_vals)
        plot_chern_vs_p(res_vals=res_vals, p_vals=p_vals)
    
    print("\n" + "="*60)
    print("ALL FIGURES GENERATED SUCCESSFULLY")
    print("="*60)
    print(f"Figures saved to: {FIGURES_DIR}")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate all figures for LaTeX summary')
    parser.add_argument('--res', type=int, default=3, help='Resolution parameter for single-res plots (default: 3)')
    parser.add_argument('--p', type=int, default=5, help='Number of layers for single-p plots (default: 5)')
    parser.add_argument('--res-vals', type=int, nargs='+', default=None, 
                       help='List of res values for multi-res plots (default: [1, 2, 3, 4])')
    parser.add_argument('--p-vals', type=int, nargs='+', default=None,
                       help='List of p values for multi-p plots (default: [2, 3, 4, 5, 6, 7, 8, 9, 10])')
    parser.add_argument('--single-only', action='store_true',
                       help='Only generate single-res plots (skip energy_vs_p_by_res and chern_vs_p)')
    
    args = parser.parse_args()
    
    main(res_val=args.res, p_val=args.p, res_vals=args.res_vals, p_vals=args.p_vals, 
         generate_all=not args.single_only)
