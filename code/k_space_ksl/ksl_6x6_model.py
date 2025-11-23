"""
KSL 6×6 Model - Implementation for the original 6-mode KSL system.

This module provides the 6×6 KSL Hamiltonian implementation following the structure
defined in main.tex precisely.

From main.tex:
- The 6x6 Hamiltonian matrix at each k is:
  H_k(t) = [[h_k, -ig(t)I, 0],
            [ig(t)I, 0, -iB(t)I],
            [0, iB(t)I, 0]]
  
- Where h_k is the 2x2 system Hamiltonian:
  h_k = [[Delta(k), if(k)],
         [-if(-k), -Delta(k)]]
  
- With:
  Delta(k) = 2*kappa*[sin(kx) - sin(ky) + sin(ky-kx)]
  f(k) = Jx + Jy*exp(-ikx) + Jz*exp(-iky)

- Fermion operators ordered as:
  [c^(z†)_(k,A), c^(z†)_(k,B), c^(y†)_(k,A), c^(y†)_(k,B), c^(x†)_(k,A), c^(x†)_(k,B)]
  
- Sublattice indices:
  0 = c^z_A, 1 = c^z_B, 2 = c^y_A, 3 = c^y_B, 4 = c^x_A, 5 = c^x_B
"""

import numpy as np
from hamiltonian_base import Hamiltonian, SingleParticleDensityMatrix


class KSLSingleParticleDensityMatrix(SingleParticleDensityMatrix):
    """
    Single-particle density matrix for KSL 6×6 model.
    
    Adds KSL-specific methods like reset_all_tau.
    """
    def __init__(self, system_size: int = None, matrix: np.ndarray = None):
        # Default system_size is 6 for KSL
        if system_size is None and matrix is None:
            system_size = 6
        super().__init__(system_size=system_size, matrix=matrix)
    
    def reset_all_tau(self) -> None:
        """
        Reset tau-related elements (sublattices 2,4 and 3,5).
        
        For KSL model, this resets the bath fermion correlations.
        Sublattices: 0=c^z_A, 1=c^z_B, 2=c^y_A, 3=c^y_B, 4=c^x_A, 5=c^x_B
        Resets correlations between (2,4) and (3,5).
        
        This follows the pattern from translational_invariant_KSL.py:
        for i in range(system_shape[0]):
            reset(2, 4, i, i)
            reset(3, 5, i, i)
        
        In our single-index system, we just call reset(2, 4) and reset(3, 5).
        """
        self.reset(2, 4)
        self.reset(3, 5)


def get_Delta_without_kappa(kx, ky):
    """
    Compute the kappa-independent part of Delta(k).
    
    This is the part that gets multiplied by kappa to get the full Delta.
    Returns: 2.0 * [sin(kx) - sin(ky) + sin(ky-kx)]
    
    To get the full Delta with kappa: kappa * get_Delta_without_kappa(kx, ky)
    """
    return 2.0 * (np.sin(kx) - np.sin(ky) + np.sin(ky - kx))


def create_KSL_hamiltonian(kx, ky, Jx=1.0, Jy=1.0, Jz=1.0, kappa=1.0, g=0.0, B=0.0):
    """
    Create KSL Hamiltonian at momentum point (kx, ky) following main.tex structure.
    
    The Hamiltonian matrix structure from main.tex Eq. (157-162):
    H_k(t) = [[h_k, -ig(t)I, 0],
              [ig(t)I, 0, -iB(t)I],
              [0, iB(t)I, 0]]
    
    Where h_k from Eq. (167-169):
    h_k = [[Delta(k), if(k)],
           [-if(-k), -Delta(k)]]
    
    Args:
        kx, ky: momentum values
        Jx, Jy, Jz: Kitaev exchange couplings (default 1.0)
        kappa: three-spin interaction strength (default 1.0)
        g: system-bath coupling (default 0.0, can be callable for time dependence)
        B: bath Zeeman field (default 0.0, can be callable for time dependence)
    
    Returns:
        hamiltonian: Hamiltonian instance
    """
    system_size = 6
    hamiltonian = Hamiltonian(system_size)
    
    # Compute Delta(k) from main.tex
    # Delta(k) = kappa * get_Delta_without_kappa(kx, ky)
    # = kappa * 2.0 * (sin(kx) - sin(ky) + sin(ky-kx))
    Delta = kappa * get_Delta_without_kappa(kx, ky)
    
    # ============================================================
    # h_k block (sublattices 0,1): system Hamiltonian
    # ============================================================
    # h_k = [[Delta(k), if(k)],
    #        [-if(-k), -Delta(k)]]
    # where f(k) = Jx + Jy*exp(-ikx) + Jz*exp(-iky) from main.tex Eq. (175)
    
    # Term 1: Delta on sublattice 0 (diagonal)
    # h_k[0,0] = Delta(k)
    hamiltonian.add_term('Delta_A', 0, 0, Delta)
    
    # Term 2: -Delta on sublattice 1 (diagonal)
    # h_k[1,1] = -Delta(k)
    hamiltonian.add_term('Delta_B', 1, 1, -Delta)
    
    # Term 3: Jx contribution to if(k) on (0,1) off-diagonal
    # h_k[0,1] contribution: i*Jx
    hamiltonian.add_term('Jx', 0, 1, 1j * Jx)
    
    # Term 4: Jy contribution to if(k) on (0,1) off-diagonal
    # h_k[0,1] contribution: i*Jy*exp(-ikx)
    hamiltonian.add_term('Jy', 0, 1, 1j * Jy * np.exp(-1j * kx))
    
    # Term 5: Jz contribution to if(k) on (0,1) off-diagonal
    # h_k[0,1] contribution: i*Jz*exp(-iky)
    hamiltonian.add_term('Jz', 0, 1, 1j * Jz * np.exp(-1j * ky))
    
    # Note: The term -if(-k) on (1,0) is automatically handled by the
    # hermitian conjugate from the off-diagonal terms
    
    # ============================================================
    # g terms: system-bath coupling
    # ============================================================
    # From main.tex: -ig(t) coupling between h_k block and c^y block
    # H_k[0,2] = -ig, H_k[1,3] = -ig
    # H_k[2,0] = ig, H_k[3,1] = ig (Hermitian conjugate)
    
    # Term 6: g_A coupling sublattice 0 (c^z_A) to sublattice 2 (c^y_A)
    # H_k[0,2] = -ig
    hamiltonian.add_term('g_A', 0, 2, -1j * g)
    
    # Term 7: g_B coupling sublattice 1 (c^z_B) to sublattice 3 (c^y_B)
    # H_k[1,3] = -ig
    hamiltonian.add_term('g_B', 1, 3, -1j * g)
    
    # ============================================================
    # B terms: bath Zeeman field
    # ============================================================
    # From main.tex: -iB(t) coupling between c^y and c^x blocks
    # H_k[2,4] = -iB, H_k[3,5] = -iB
    # H_k[4,2] = iB, H_k[5,3] = iB (Hermitian conjugate)
    
    # Term 8: B_A coupling sublattice 2 (c^y_A) to sublattice 4 (c^x_A)
    # H_k[2,4] = -iB
    hamiltonian.add_term('B_A', 2, 4, -1j * B)
    
    # Term 9: B_B coupling sublattice 3 (c^y_B) to sublattice 5 (c^x_B)
    # H_k[3,5] = -iB
    hamiltonian.add_term('B_B', 3, 5, -1j * B)
    
    return hamiltonian

