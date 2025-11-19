"""
K-space KSL Model - Self-contained implementation.

This module provides a completely self-contained implementation of the KSL Hamiltonian
following the structure defined in main.tex precisely.

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
from scipy.linalg import expm, eigh
from scipy import sparse
from typing import Union


class HamiltonianTerm:
    """
    Represents a fermion bilinear term: strength*c^dagger_i*c_j + h.c.
    
    For i == j: Only diagonal term (no hermitian conjugate needed)
    For i != j: Off-diagonal term with hermitian conjugate
    """
    
    def __init__(self, i: int, j: int, strength: Union[float, complex], system_size: int):
        """
        Initialize a Hamiltonian term.
        
        Args:
            i: First fermion mode index (0 to system_size-1)
            j: Second fermion mode index (0 to system_size-1)
            strength: Coupling strength (float or complex)
            system_size: Total number of fermion modes (default 6 for KSL)
        """
        if not (0 <= i < system_size and 0 <= j < system_size):
            raise ValueError(f"Indices i={i}, j={j} must be in range [0, {system_size})")
        
        self.i = i
        self.j = j
        self.strength = strength
        self.system_size = system_size
    
    def get_matrix(self) -> np.ndarray:
        """
        Get the matrix representation of this term.
        
        Returns:
            system_size x system_size numpy array
            - If i == j: H[i,i] = strength (diagonal)
            - If i != j: H[i,j] = strength, H[j,i] = conj(strength) (off-diagonal with h.c.)
        """
        H = np.zeros((self.system_size, self.system_size), dtype=complex)
        
        if self.i == self.j:
            # Diagonal case: no hermitian conjugate needed
            H[self.i, self.i] = self.strength
        else:
            # Off-diagonal case: add hermitian conjugate
            H[self.i, self.j] = self.strength
            H[self.j, self.i] = np.conj(self.strength)
        
        return H
    
    def exp_unitary(self, theta: float) -> sparse.csr_matrix:
        """
        Compute exp(-i*theta*strength/abs(strength)*H_term).
        
        Args:
            theta: Rotation angle
            
        Returns:
            system_size x system_size sparse CSR unitary matrix
        """
        # Handle zero strength case
        if np.abs(self.strength) < 1e-10:
            raise ValueError(f"Strength of term at {self.i}, {self.j} is zero")
        
        # Normalize strength
        strength_normalized = self.strength / np.abs(self.strength)
        
        # Create sparse identity matrix
        U = sparse.eye(self.system_size, format='lil', dtype=complex)
        
        if self.i == self.j:
            # Diagonal case: exp(-i*theta*strength_normalized) on diagonal
            U[self.i, self.i] = np.exp(-1j * theta * strength_normalized)
        else:
            # Off-diagonal case: 2x2 block
            H_2x2 = np.array([
                [0, strength_normalized],
                [np.conj(strength_normalized), 0]
            ], dtype=complex)
            
            # Compute exp(-i*theta*H_2x2)
            U_2x2 = expm(-1j * theta * H_2x2)
            
            # Embed in full matrix
            U[np.ix_([self.i, self.j], [self.i, self.j])] = U_2x2
        
        # Convert to CSR format for efficient multiplication
        return U.tocsr()


class Hamiltonian:
    """
    Collection of Hamiltonian terms forming a complete Hamiltonian.
    """
    
    def __init__(self, system_size: int = 6):
        """
        Initialize an empty Hamiltonian.
        
        Args:
            system_size: Total number of fermion modes (default 6 for KSL)
        """
        self.system_size = system_size
        self.terms = {}
    
    def add_term(self, name: str, i: int, j: int, strength: Union[float, complex]):
        """
        Add a Hamiltonian term.
        
        Args:
            name: Name identifier for the term
            i: First fermion mode index
            j: Second fermion mode index
            strength: Coupling strength
        """
        self.terms[name] = HamiltonianTerm(i, j, strength, self.system_size)
    
    def get_matrix(self) -> np.ndarray:
        """
        Get the full Hamiltonian matrix (sum of all terms).
        
        Returns:
            system_size x system_size numpy array
        """
        H = np.zeros((self.system_size, self.system_size), dtype=complex)
        for term in self.terms.values():
            H += term.get_matrix()
        return H
    
    def get_ground_state(self) -> np.ndarray:
        """
        Compute the ground state single-particle density matrix.
        
        Returns:
            system_size x system_size density matrix (projection onto negative energy states)
        """
        M = self.get_matrix()
        e, Q = eigh(M)
        S = Q.conj() @ np.diag(e < 0) @ Q.T
        return S
    
    def compute_energy(self, state: np.ndarray) -> float:
        """
        Compute the energy of a state given this Hamiltonian.
        
        Args:
            state: Density matrix (single-particle), system_size x system_size
        
        Returns:
            Energy value
        """
        H_matrix = self.get_matrix()
        # Energy = Tr(S @ H)
        energy = np.trace(state @ H_matrix.T).real
        return energy


class VariationalCircuit:
    """
    Variational circuit constructed from Hamiltonian terms and parameter values.
    
    Parameters are stored as a dictionary mapping term names to lists of parameter
    values, where each list has one value per circuit layer.
    """
    
    def __init__(self, hamiltonian: Hamiltonian, parameters: dict[str, list[float]], num_layers: int = None):
        """
        Initialize variational circuit.
        
        Args:
            hamiltonian: Hamiltonian instance containing the terms
            parameters: Dictionary mapping term names to lists of parameter values.
                       Each list should have the same length (number of layers).
            num_layers: Optional number of layers. If None, inferred from parameter lists.
        """
        self.hamiltonian = hamiltonian
        
        if not parameters:
            raise ValueError("parameters dictionary cannot be empty")
        
        # Infer number of layers from first parameter list
        first_key = next(iter(parameters))
        inferred_num_layers = len(parameters[first_key])
        
        if num_layers is None:
            num_layers = inferred_num_layers
        else:
            if inferred_num_layers != num_layers:
                raise ValueError(f"Parameter list length {inferred_num_layers} doesn't match num_layers {num_layers}")
        
        # Validate all parameter lists have the same length
        for name, param_list in parameters.items():
            if len(param_list) != num_layers:
                raise ValueError(f"Parameter list for '{name}' has length {len(param_list)}, expected {num_layers}")
        
        self.parameters = parameters
        self.num_layers = num_layers
    
    def get_unitary(self, term_order: list[str] = None) -> np.ndarray:
        """
        Construct the full variational circuit unitary.
        
        The circuit is constructed as a product of layers, where each layer applies
        unitaries from each Hamiltonian term in the specified order.
        
        Args:
            term_order: Optional list specifying the order to apply terms.
                       If None, uses the order terms appear in hamiltonian.terms
        
        Returns:
            system_size x system_size dense unitary matrix
        """
        if term_order is None:
            # Use the order terms appear in the Hamiltonian
            term_order = list(self.hamiltonian.terms.keys())
        
        # Validate all terms in term_order exist
        for term_name in term_order:
            if term_name not in self.hamiltonian.terms:
                raise ValueError(f"Term '{term_name}' not found in Hamiltonian")
            if term_name not in self.parameters:
                raise ValueError(f"Parameters for term '{term_name}' not found")
        
        # Initialize unitary as identity (sparse for efficiency)
        system_size = self.hamiltonian.system_size
        Ud = sparse.eye(system_size, format='csr', dtype=complex)
        
        # Apply each layer
        for layer in range(self.num_layers):
            # Apply each term in the specified order
            for term_name in term_order:
                theta = self.parameters[term_name][layer]
                term = self.hamiltonian.terms[term_name]
                U_term = term.exp_unitary(theta)
                # Multiply unitaries (sparse matrix multiplication)
                Ud = U_term @ Ud
        
        # Convert to dense matrix for return
        return Ud.toarray()


class SingleParticleDensityMatrix:
    """
    Single-particle density matrix for fermionic systems.
    
    Represents the expectation value <c^dagger_i c_j>.
    """
    
    def __init__(self, system_size: int = None, matrix: np.ndarray = None):
        """
        Initialize a single-particle density matrix.
        
        Args:
            system_size: Number of fermion modes (required if matrix is None)
            matrix: Optional initial density matrix (system_size x system_size)
                   If provided, system_size is inferred from matrix shape
        """
        if matrix is not None:
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError(f"Matrix must be square, got shape {matrix.shape}")
            self.system_size = matrix.shape[0]
            self.matrix = matrix.copy()
        elif system_size is not None:
            self.system_size = system_size
            self.matrix = np.zeros((system_size, system_size), dtype=complex)
        else:
            raise ValueError("Either system_size or matrix must be provided")
    
    def evolve_state_with_unitary(self, Ud: np.ndarray) -> None:
        """
        Evolve the state with a unitary: S -> Ud @ S @ Ud^dagger.
        
        Args:
            Ud: Unitary matrix (system_size x system_size)
        """
        if Ud.shape != (self.system_size, self.system_size):
            raise ValueError(f"Unitary shape {Ud.shape} doesn't match system_size {self.system_size}")
        self.matrix = Ud.conj() @ self.matrix @ Ud.T
    
    def initialize(self, initial_state_type: str = 'product'):
        """
        Initialize the density matrix.
        
        Args:
            initial_state_type: Type of initial state
                - 'full': Product state with all c^z fermions occupied (identity matrix)
                - 'random': Random half-filled state (0.5 * identity)
                - 'empty': Empty state (zero matrix)
        """
        if initial_state_type == 'full':
            # Product state: all c^z fermions occupied
            self.matrix = np.eye(self.system_size, dtype=complex)
        elif initial_state_type == 'random':
            # Random state: half-filled
            self.matrix = 0.5 * np.eye(self.system_size, dtype=complex)
        elif initial_state_type == 'empty':
            # Empty state
            self.matrix = np.zeros((self.system_size, self.system_size), dtype=complex)
        else:
            raise ValueError(f"Unknown initial_state_type: {initial_state_type}")


class KSLSingleParticleDensityMatrix(SingleParticleDensityMatrix):
    """
    Single-particle density matrix for KSL model.
    
    Adds KSL-specific methods like reset_all_tau.
    """
    def __init__(self, system_size: int = None, matrix: np.ndarray = None):
        # Default system_size is 6 for KSL
        if system_size is None and matrix is None:
            system_size = 6
        super().__init__(system_size=system_size, matrix=matrix)
    
    def reset(self, i: int, j: int) -> None:
        """
        Reset correlations between modes i and j.
        
        Sets the 2x2 block to [[0.5, 0.5*1j], [-0.5*1j, 0.5]].
        This is similar to the reset method in ComplexSingleParticleDensityMatrix.
        
        Args:
            i: First mode index
            j: Second mode index
        """
        # Zero out all correlations involving these modes
        self.matrix[:, [i, j]] = 0
        self.matrix[[i, j], :] = 0
        # Set the 2x2 block to the reset value
        self.matrix[np.ix_([i, j], [i, j])] = np.array([[0.5, -0.5*1j], [0.5*1j, 0.5]], dtype=complex)
    
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

