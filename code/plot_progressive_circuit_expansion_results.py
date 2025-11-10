import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path to import from variational_circuit_KSL_numba
sys.path.insert(0, os.path.dirname(__file__))

# Data directory path (relative to this file)
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')

# Import functions from variational_circuit_KSL_numba
from variational_circuit_KSL_numba import (
    load_optimized_parameters_for_res_p,
    simulate_grid_with_analysis,
    n_cycles_test,
    n_cycles_train
)


def main(csv_path: str = None) -> None:
    if csv_path is None:
        csv_path = os.path.join(DATA_DIR, "progressive_circuit_expansion_results.csv")
    df = pd.read_csv(csv_path)

    # Use test energy for plots; drop failures
    df = df.dropna(subset=["energy_density_test"])  # keep completed rows

    res_vals = sorted(df["res"].unique())
    p_vals = sorted(df["p"].unique())

    # 1) Energy vs p (one line per res): solid=test, dashed=train (same color)
    plt.figure(figsize=(7, 4))
    for res in res_vals:
        dd = df[df.res == res].sort_values("p")
        # Plot test (solid)
        line_test, = plt.plot(dd.p, dd.energy_density_test, "-o", label=f"{res*6}", linewidth=2)
        color = line_test.get_color()
        # Plot train (dashed) with same color
        plt.plot(dd.p, dd.energy_density_train, "--o", color=color, alpha=0.8, linewidth=1.5)
    
    plt.xlabel("p", fontsize=20)
    plt.ylabel("Energy density", fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.legend(title="$L_{\\text{train}}$", title_fontsize=20, fontsize=17, ncol=2)
    plt.tick_params(labelsize=17)
    plt.tight_layout()
    
    # Save the figure BEFORE showing it
    fig_path = os.path.join(os.path.dirname(__file__), "../figures/energy_vs_p_by_res.pdf")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved energy_vs_p_by_res.pdf to {fig_path}")
    
    # Print best (test) configuration
    best = df.loc[df["energy_density_test"].idxmin()]
    print(
        "Best (test):",
        {
            "res": int(best["res"]),
            "p": int(best["p"]),
            "n_k_points_train": int(best["n_k_points_train"]),
            "energy_density_train": float(best["energy_density_train"]),
            "energy_density_test": float(best["energy_density_test"]),
        },
    )
    
    # Show the figure (optional, can be removed for headless operation)
    # plt.show()
    plt.close()
    
    # 2) Chern number vs p
    plot_chern_vs_p(df, res_vals)

def plot_chern_vs_p(df, res_vals=None):
    """
    Generate chern_vs_p.pdf - Finite size spectral Chern number (ν) vs. circuit depth p
    
    Args:
        df: DataFrame with results from progressive circuit expansion
        res_vals: list of res values to plot (default: use all in df)
    """
    import variational_circuit_KSL_numba as vc_module
    
    print("\n" + "="*60)
    print("Generating chern_vs_p.pdf")
    print("="*60)
    
    if res_vals is None:
        res_vals = sorted(df["res"].unique())
    
    # Create test grid (fixed for all res values)
    n_k_points_test = 6 * 20  # 120
    # include the endpoint because of finite derivative for chern number
    kx_list_test = np.linspace(-np.pi, np.pi, n_k_points_test + 1)
    ky_list_test = np.linspace(-np.pi, np.pi, n_k_points_test + 1)
    
    plt.figure(figsize=(7, 4))
    
    # Plot for each res value (similar to energy_vs_p_by_res)
    for res in res_vals:
        print(f"Processing res={res}...")
        
        # Get p values for this res
        df_res = df[df.res == res].sort_values("p")
        p_vals = sorted(df_res["p"].unique())
        
        # Calculate Chern numbers for training and test grids
        system_chern_train = []
        system_chern_test = []
        p_vals_valid = []
        
        # Create training grid for this res
        n_k_points_train = 6 * res
        # include the endpoint because of finite derivative for chern number
        kx_list_train = np.linspace(-np.pi, np.pi, n_k_points_train + 1)
        ky_list_train = np.linspace(-np.pi, np.pi, n_k_points_train + 1)
        
        for p_val in p_vals:
            print(f"  Evaluating p={p_val}...")
            
            try:
                # Load parameters
                strength_durations = load_optimized_parameters_for_res_p(res, p_val)
                
                # Set global p
                vc_module.p = p_val
                
                # Run simulation on training grid (use training cycles to match energy evaluation)
                _, _, _, system_chern_tr, _, _ = simulate_grid_with_analysis(
                    kx_list_train, ky_list_train, strength_durations, n_cycles=n_cycles_train
                )
                
                # Run simulation on test grid (use test cycles to match energy evaluation)
                _, _, _, system_chern_te, _, _ = simulate_grid_with_analysis(
                    kx_list_test, ky_list_test, strength_durations, n_cycles=n_cycles_test
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
    
    plt.xlabel("p", fontsize=20)
    plt.ylabel("$\\nu$", fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.legend(title="$L_{\\text{train}}$", title_fontsize=20, fontsize=17, ncol=2)
    plt.tick_params(labelsize=17)
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(os.path.dirname(__file__), "../figures/chern_vs_p.pdf")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved chern_vs_p.pdf to {fig_path}")


if __name__ == "__main__":
    main()


