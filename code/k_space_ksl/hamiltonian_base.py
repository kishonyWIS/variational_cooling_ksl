"""
Base Hamiltonian Classes - Generic implementation.

This module provides generic, system-size agnostic classes for constructing
Hamiltonians, variational circuits, and density matrices for fermionic systems.
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
            system_size: Total number of fermion modes
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
        Compute exp(-i*theta*H_term) where H_term uses full self.strength.
        
        Args:
            theta: Real prefactor (rotation angle/time step)
            
        Returns:
            system_size x system_size sparse CSR unitary matrix
        """
        # Handle zero strength case: return identity
        if np.abs(self.strength) < 1e-10:
            return sparse.eye(self.system_size, format='csr', dtype=complex)
        
        # Handle zero theta case: return identity
        if np.abs(theta) < 1e-10:
            return sparse.eye(self.system_size, format='csr', dtype=complex)
        
        # Create sparse identity matrix
        U = sparse.eye(self.system_size, format='lil', dtype=complex)
        
        if self.i == self.j:
            # Diagonal case: exp(-i*theta*strength) on diagonal
            U[self.i, self.i] = np.exp(-1j * theta * self.strength)
        else:
            # Off-diagonal case: 2x2 block
            H_2x2 = np.array([
                [0, self.strength],
                [np.conj(self.strength), 0]
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
    
    def __init__(self, system_size: int):
        """
        Initialize an empty Hamiltonian.
        
        Args:
            system_size: Total number of fermion modes
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
    
    def reset(self, i: int, j: int) -> None:
        """
        Reset correlations between modes i and j.
        
        Sets the 2x2 block to [[0.5, 0.5*1j], [-0.5*1j, 0.5]].
        This is a generic method that works for any fermionic system.
        
        Args:
            i: First mode index
            j: Second mode index
        """
        # Zero out all correlations involving these modes
        self.matrix[:, [i, j]] = 0
        self.matrix[[i, j], :] = 0
        # Set the 2x2 block to the reset value
        self.matrix[np.ix_([i, j], [i, j])] = np.array([[0.5, -0.5*1j], [0.5*1j, 0.5]], dtype=complex)

