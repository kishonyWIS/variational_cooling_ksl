"""
Script to multiply all optimized parameters by 2.
This is needed after refactoring to make all parameters unitless and periodic mod 2π.
"""

import numpy as np
import os
import glob

# Data directory path
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
OPTIMIZED_PARAMS_DIR = os.path.join(DATA_DIR, 'optimized_parameters')

def multiply_parameters_by_2():
    """
    Multiply all parameters in all .npz files by 2.
    This updates the parameters to be consistent with the new unitless parameterization.
    """
    # Find all .npz files in the optimized_parameters directory
    pattern = os.path.join(OPTIMIZED_PARAMS_DIR, 'optimized_params_*.npz')
    files = glob.glob(pattern)
    
    if not files:
        print(f"No parameter files found in {OPTIMIZED_PARAMS_DIR}")
        return
    
    print(f"Found {len(files)} parameter files to update")
    print("="*60)
    
    for filepath in sorted(files):
        filename = os.path.basename(filepath)
        print(f"Processing {filename}...")
        
        # Load the parameters
        data = np.load(filepath)
        strength_durations = {}
        for term in ['Jx', 'Jy', 'Jz', 'kappa', 'g', 'B']:
            if term in data:
                strength_durations[term] = data[term] * 2.0
            else:
                print(f"  Warning: {term} not found in {filename}")
        
        # Save the updated parameters
        np.savez(filepath, **strength_durations)
        print(f"  Updated: all parameters multiplied by 2")
    
    print("="*60)
    print(f"Successfully updated {len(files)} parameter files")
    print("All parameters have been multiplied by 2")

if __name__ == "__main__":
    multiply_parameters_by_2()

