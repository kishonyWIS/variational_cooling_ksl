"""
KSL 24×24 Model - Implementation for the 24-mode KSL system with 2×2 supercell.

This module provides the 24×24 KSL Hamiltonian implementation following the structure
defined in decomposing_kappa.tex.

The system uses a 2×2 supercell with 4 internal offsets (r₀, r₁, r₂, r₃).
Total modes: 2 sublattices × 3 Majorana types × 4 offsets = 24 modes

Mode ordering:
- c^z fermions: modes 0-7 (A_r0, A_r1, A_r2, A_r3, B_r0, B_r1, B_r2, B_r3)
- c^y fermions: modes 8-15 (same pattern)
- c^x fermions: modes 16-23 (same pattern)

The 8×8 system Hamiltonian block (for c^z fermions) follows decomposing_kappa.tex exactly.
"""

import numpy as np
from hamiltonian_base import Hamiltonian, SingleParticleDensityMatrix


def convert_k_to_K(kx, ky):
    """
    Convert k-space coordinates to K-space (Bloch wavevector for supercell).
    
    From decomposing_kappa.tex:
    K·a = 2K_i, K·b = 2K_j
    where a = 2i, b = 2j, so K_i = kx/2, K_j = ky/2
    
    Args:
        kx, ky: Original momentum space coordinates
    
    Returns:
        tuple: (K_i, K_j) Bloch wavevector components
    """
    return kx / 2.0, ky / 2.0


def get_kappa_term_name(S, d, p, component):
    """
    Generate a name for a kappa term.
    
    Args:
        S: Sublattice class ('A' or 'B')
        d: Direction ('+i', '-j', 'j-i')
        p: Parity ('even' or 'odd')
        component: Component ('first' or 'second') - distinguishes the two terms in each block
    
    Returns:
        str: Term name, e.g., 'kappa_A_+i_even_first'
    """
    # Normalize direction string
    d_str = d.replace('+', 'p').replace('-', 'm').replace('j-i', 'jmi')
    return f'kappa_{S}_{d_str}_{p}_{component}'


def get_all_kappa_term_names():
    """
    Get all 24 kappa term names.
    
    Each of the 12 blocks (S, d, p) has 2 components: 'first' and 'second'
    for the two pairs of modes within that block.
    
    Returns:
        list: List of 24 kappa term names
    """
    names = []
    for S in ['A', 'B']:
        for d in ['+i', '-j', 'j-i']:
            for p in ['even', 'odd']:
                for component in ['first', 'second']:
                    names.append(get_kappa_term_name(S, d, p, component))
    return names


class KSL24SingleParticleDensityMatrix(SingleParticleDensityMatrix):
    """
    Single-particle density matrix for KSL 24×24 model.
    
    Adds KSL-specific methods like reset_all_tau generalized to all 4 offsets.
    """
    def __init__(self, system_size: int = None, matrix: np.ndarray = None):
        # Default system_size is 24 for KSL 24×24
        if system_size is None and matrix is None:
            system_size = 24
        super().__init__(system_size=system_size, matrix=matrix)
    
    def reset_all_tau(self) -> None:
        """
        Reset tau-related elements for all 4 internal offsets.
        
        For KSL 24×24 model, this resets the bath fermion correlations.
        Mode structure:
        - c^z: modes 0-7 (A_r0-r3, B_r0-r3)
        - c^y: modes 8-15 (A_r0-r3, B_r0-r3)
        - c^x: modes 16-23 (A_r0-r3, B_r0-r3)
        
        Resets correlations between (c^y_A_r, c^x_A_r) and (c^y_B_r, c^x_B_r) for r ∈ {0,1,2,3}.
        Total: 8 reset operations (4 offsets × 2 sublattices)
        """
        # For each offset r ∈ {0,1,2,3}:
        # - A sublattice: reset between c^y (mode 8+r) and c^x (mode 16+r)
        # - B sublattice: reset between c^y (mode 12+r) and c^x (mode 20+r)
        for r in range(4):
            # A sublattice: c^y_A_r <-> c^x_A_r
            self.reset(8 + r, 16 + r)
            # B sublattice: c^y_B_r <-> c^x_B_r
            self.reset(12 + r, 20 + r)


def create_KSL_24x24_hamiltonian(K_i, K_j, Jx=1.0, Jy=1.0, Jz=1.0, kappa=1.0, g=0.0, B=0.0):
    """
    Create KSL 24×24 Hamiltonian at Bloch wavevector (K_i, K_j) following decomposing_kappa.tex.
    
    The Hamiltonian has a 24×24 structure with:
    - 8×8 system block for c^z fermions (modes 0-7)
    - 8×8 block for c^y fermions (modes 8-15)
    - 8×8 block for c^x fermions (modes 16-23)
    - Couplings between blocks via g and B terms
    
    Args:
        K_i, K_j: Bloch wavevector components (not kx, ky - use convert_k_to_K to convert)
        Jx, Jy, Jz: Kitaev exchange couplings (default 1.0)
        kappa: three-spin interaction strength (default 1.0)
              This will be distributed to 24 separate kappa terms in the variational circuit
        g: system-bath coupling (default 0.0)
        B: bath Zeeman field (default 0.0)
    
    Returns:
        hamiltonian: Hamiltonian instance with system_size=24
    """
    system_size = 24
    hamiltonian = Hamiltonian(system_size)
    
    # ============================================================
    # Mode indices for clarity:
    # c^z: 0-7 (A_r0=0, A_r1=1, A_r2=2, A_r3=3, B_r0=4, B_r1=5, B_r2=6, B_r3=7)
    # c^y: 8-15 (A_r0=8, A_r1=9, A_r2=10, A_r3=11, B_r0=12, B_r1=13, B_r2=14, B_r3=15)
    # c^x: 16-23 (A_r0=16, A_r1=17, A_r2=18, A_r3=19, B_r0=20, B_r1=21, B_r2=22, B_r3=23)
    # ============================================================
    
    # ============================================================
    # J terms: Nearest-neighbor Kitaev interactions (A--B block)
    # Following decomposing_kappa.tex lines 115-120
    # ============================================================
    
    # J_x terms: H_AB[r, r] = iJ_x for all r ∈ {0,1,2,3}
    for r in range(4):
        # A_r to B_r coupling in c^z block
        hamiltonian.add_term(f'Jx_r{r}', r, 4 + r, 1j * Jx)
    
    # J_y terms: Following decomposing_kappa.tex line 118
    # H_AB[i,j] where i is A-sublattice index (0-3) and j is B-sublattice index (0-3)
    # In full 8×8 matrix: A modes are 0-3, B modes are 4-7
    # H_AB[1,0] = iJ_y means A_r1 (mode 1) -> B_r0 (mode 4)
    # H_AB[0,1] = iJ_y*e^{+2iK_i} means A_r0 (mode 0) -> B_r1 (mode 5)
    hamiltonian.add_term('Jy_A1_B0', 1, 4, 1j * Jy)  # A_r1 -> B_r0
    hamiltonian.add_term('Jy_A0_B1', 0, 5, 1j * Jy * np.exp(+2j * K_i))  # A_r0 -> B_r1 (with phase)
    # H_AB[3,2] = iJ_y means A_r3 (mode 3) -> B_r2 (mode 6)
    # H_AB[2,3] = iJ_y*e^{+2iK_i} means A_r2 (mode 2) -> B_r3 (mode 7)
    hamiltonian.add_term('Jy_A3_B2', 3, 6, 1j * Jy)  # A_r3 -> B_r2
    hamiltonian.add_term('Jy_A2_B3', 2, 7, 1j * Jy * np.exp(+2j * K_i))  # A_r2 -> B_r3 (with phase)
    
    # J_z terms: Following decomposing_kappa.tex line 119
    # H_AB[2,0] = iJ_z means A_r2 (mode 2) -> B_r0 (mode 4)
    # H_AB[0,2] = iJ_z*e^{+2iK_j} means A_r0 (mode 0) -> B_r2 (mode 6)
    hamiltonian.add_term('Jz_A2_B0', 2, 4, 1j * Jz)  # A_r2 -> B_r0
    hamiltonian.add_term('Jz_A0_B2', 0, 6, 1j * Jz * np.exp(+2j * K_j))  # A_r0 -> B_r2 (with phase)
    # H_AB[3,1] = iJ_z means A_r3 (mode 3) -> B_r1 (mode 5)
    # H_AB[1,3] = iJ_z*e^{+2iK_j} means A_r1 (mode 1) -> B_r3 (mode 7)
    hamiltonian.add_term('Jz_A3_B1', 3, 5, 1j * Jz)  # A_r3 -> B_r1
    hamiltonian.add_term('Jz_A1_B3', 1, 7, 1j * Jz * np.exp(+2j * K_j))  # A_r1 -> B_r3 (with phase)
    
    # ============================================================
    # Kappa terms: 24 terms (12 blocks × 2 components)
    # Following decomposing_kappa.tex lines 93-113
    # ============================================================
    
    # For now, we'll use a single kappa value that will be distributed
    # to 24 separate terms in the variational circuit.
    # Here we create all 24 terms with the same base strength.
    
    # Each of the 12 blocks contributes to specific matrix elements.
    # We create 24 separate terms (12 blocks × 2 components: even/odd)
    # Each term gets a unique name for variational control.
    
    # A--A block kappa terms (lines 94-101)
    # Each block has two components: "first" and "second" for the two pairs of modes
    # H_κ^{A,+i,even} = i*κ_{A,+i,e}*(c_{r0}c_{r1} + c_{r2}c_{r3})
    #   - first component: c_{r0}c_{r1} → [0,1]
    #   - second component: c_{r2}c_{r3} → [2,3]
    
    # Direction +i, even block
    hamiltonian.add_term(get_kappa_term_name('A', '+i', 'even', 'first'), 0, 1, 1j * kappa)  # r0-r1 pair
    hamiltonian.add_term(get_kappa_term_name('A', '+i', 'even', 'second'), 2, 3, 1j * kappa)  # r2-r3 pair
    # Direction +i, odd block
    hamiltonian.add_term(get_kappa_term_name('A', '+i', 'odd', 'first'), 0, 1, -1j * kappa * np.exp(-2j * K_i))  # r0-r1 pair
    hamiltonian.add_term(get_kappa_term_name('A', '+i', 'odd', 'second'), 2, 3, -1j * kappa * np.exp(-2j * K_i))  # r2-r3 pair
    
    # Direction -j, even block: H_κ^{A,-j,even} = -i*κ_{A,-j,e}*(c_{r0}c_{r2} + c_{r1}c_{r3})
    #   - first component: c_{r0}c_{r2} → [0,2]
    #   - second component: c_{r1}c_{r3} → [1,3]
    hamiltonian.add_term(get_kappa_term_name('A', '-j', 'even', 'first'), 0, 2, -1j * kappa)  # r0-r2 pair
    hamiltonian.add_term(get_kappa_term_name('A', '-j', 'even', 'second'), 1, 3, -1j * kappa)  # r1-r3 pair
    # Direction -j, odd block
    hamiltonian.add_term(get_kappa_term_name('A', '-j', 'odd', 'first'), 0, 2, 1j * kappa * np.exp(-2j * K_j))  # r0-r2 pair
    hamiltonian.add_term(get_kappa_term_name('A', '-j', 'odd', 'second'), 1, 3, 1j * kappa * np.exp(-2j * K_j))  # r1-r3 pair
    
    # Direction j-i, even block: H_κ^{A,j-i,even} = i*κ_{A,j-i,e}*(c_{r1}c_{r2} + c_{r3}c_{r0+})
    #   - first component: c_{r1}c_{r2} → [1,2]
    #   - second component: c_{r3}c_{r0+} → [3,0] with phase e^{+2iK_j}
    hamiltonian.add_term(get_kappa_term_name('A', 'j-i', 'even', 'first'), 1, 2, 1j * kappa)  # r1-r2 pair
    hamiltonian.add_term(get_kappa_term_name('A', 'j-i', 'even', 'second'), 3, 0, 1j * kappa * np.exp(+2j * K_j))  # r3-r0 pair
    # Direction j-i, odd block
    hamiltonian.add_term(get_kappa_term_name('A', 'j-i', 'odd', 'first'), 1, 2, -1j * kappa * np.exp(+2j * (K_i - K_j)))  # r1-r2 pair
    hamiltonian.add_term(get_kappa_term_name('A', 'j-i', 'odd', 'second'), 3, 0, -1j * kappa * np.exp(+2j * K_i))  # r3-r0 pair
    
    # B--B block kappa terms (lines 104-112) - opposite signs
    # Direction +i, even block (opposite sign from A)
    hamiltonian.add_term(get_kappa_term_name('B', '+i', 'even', 'first'), 4, 5, -1j * kappa)  # B_r0-B_r1 pair
    hamiltonian.add_term(get_kappa_term_name('B', '+i', 'even', 'second'), 6, 7, -1j * kappa)  # B_r2-B_r3 pair
    # Direction +i, odd block
    hamiltonian.add_term(get_kappa_term_name('B', '+i', 'odd', 'first'), 4, 5, 1j * kappa * np.exp(-2j * K_i))  # B_r0-B_r1 pair
    hamiltonian.add_term(get_kappa_term_name('B', '+i', 'odd', 'second'), 6, 7, 1j * kappa * np.exp(-2j * K_i))  # B_r2-B_r3 pair
    
    # Direction -j, even block (opposite sign from A)
    hamiltonian.add_term(get_kappa_term_name('B', '-j', 'even', 'first'), 4, 6, 1j * kappa)  # B_r0-B_r2 pair
    hamiltonian.add_term(get_kappa_term_name('B', '-j', 'even', 'second'), 5, 7, 1j * kappa)  # B_r1-B_r3 pair
    # Direction -j, odd block
    hamiltonian.add_term(get_kappa_term_name('B', '-j', 'odd', 'first'), 4, 6, -1j * kappa * np.exp(-2j * K_j))  # B_r0-B_r2 pair
    hamiltonian.add_term(get_kappa_term_name('B', '-j', 'odd', 'second'), 5, 7, -1j * kappa * np.exp(-2j * K_j))  # B_r1-B_r3 pair
    
    # Direction j-i, even block (opposite sign from A)
    hamiltonian.add_term(get_kappa_term_name('B', 'j-i', 'even', 'first'), 5, 6, -1j * kappa)  # B_r1-B_r2 pair
    hamiltonian.add_term(get_kappa_term_name('B', 'j-i', 'even', 'second'), 7, 4, -1j * kappa * np.exp(+2j * K_j))  # B_r3-B_r0 pair
    # Direction j-i, odd block
    hamiltonian.add_term(get_kappa_term_name('B', 'j-i', 'odd', 'first'), 5, 6, 1j * kappa * np.exp(+2j * (K_i - K_j)))  # B_r1-B_r2 pair
    hamiltonian.add_term(get_kappa_term_name('B', 'j-i', 'odd', 'second'), 7, 4, 1j * kappa * np.exp(+2j * K_i))  # B_r3-B_r0 pair
    
    # ============================================================
    # g terms: system-bath coupling generalized to all 4 offsets
    # ============================================================
    # g_A: Couplings between c^z_A_r and c^y_A_r for r ∈ {0,1,2,3}
    # g_B: Couplings between c^z_B_r and c^y_B_r for r ∈ {0,1,2,3}
    for r in range(4):
        # A sublattice: c^z_A_r (mode r) <-> c^y_A_r (mode 8+r)
        hamiltonian.add_term(f'g_A_r{r}', r, 8 + r, -1j * g)
        # B sublattice: c^z_B_r (mode 4+r) <-> c^y_B_r (mode 12+r)
        hamiltonian.add_term(f'g_B_r{r}', 4 + r, 12 + r, -1j * g)
    
    # ============================================================
    # B terms: bath Zeeman field generalized to all 4 offsets
    # ============================================================
    # B_A: Couplings between c^y_A_r and c^x_A_r for r ∈ {0,1,2,3}
    # B_B: Couplings between c^y_B_r and c^x_B_r for r ∈ {0,1,2,3}
    for r in range(4):
        # A sublattice: c^y_A_r (mode 8+r) <-> c^x_A_r (mode 16+r)
        hamiltonian.add_term(f'B_A_r{r}', 8 + r, 16 + r, -1j * B)
        # B sublattice: c^y_B_r (mode 12+r) <-> c^x_B_r (mode 20+r)
        hamiltonian.add_term(f'B_B_r{r}', 12 + r, 20 + r, -1j * B)
    
    return hamiltonian

