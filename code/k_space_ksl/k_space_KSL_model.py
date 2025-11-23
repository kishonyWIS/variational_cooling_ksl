"""
K-space KSL Model - Backward compatibility wrapper.

This module maintains backward compatibility by re-exporting classes and functions
from the refactored modules: hamiltonian_base.py and ksl_6x6_model.py
"""

# Import from base module
from hamiltonian_base import (
    HamiltonianTerm,
    Hamiltonian,
    VariationalCircuit,
    SingleParticleDensityMatrix
)

# Import from 6x6 model
from ksl_6x6_model import (
    KSLSingleParticleDensityMatrix,
    get_Delta_without_kappa,
    create_KSL_hamiltonian
)

# Re-export everything for backward compatibility
__all__ = [
    'HamiltonianTerm',
    'Hamiltonian',
    'VariationalCircuit',
    'SingleParticleDensityMatrix',
    'KSLSingleParticleDensityMatrix',
    'get_Delta_without_kappa',
    'create_KSL_hamiltonian'
]
