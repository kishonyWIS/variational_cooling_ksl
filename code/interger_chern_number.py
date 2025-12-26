import numpy as np
from itertools import product

def chern_from_mixed_spdm(R):
    """
    Compute a continuous (non-quantized) 'Chern number' for a mixed
    single-particle density matrix R(k).

    This uses the plaquette phase formula:
        Φ_□(i,j) = arg[ Tr( R₀ R₁ R₂ R₃ ) ]
    where R₀,R₁,R₂,R₃ are the SPDMs at the four corners of the
    plaquette (i,j) → (i+1,j) → (i+1,j+1) → (i,j+1).

    Parameters
    ----------
    R : ndarray, shape (Nx, Ny, M, M)
        Hermitian positive semidefinite matrices with eigenvalues ∈ [0,1].
        Need not be projectors or normalized.

    Returns
    -------
    Chern_cont : float
        The non-integer mixed-state 'Chern' value = ΣΦ/(2π).
    """
    assert R.ndim == 4 and R.shape[2] == R.shape[3], "Expected (Nx,Ny,M,M)."
    Nx, Ny, M, _ = R.shape

    def wrapx(a): return a % Nx
    def wrapy(b): return b % Ny

    flux = np.zeros((Nx, Ny), dtype=float)
    total_phase = 0.0
    for i in range(Nx):
        for j in range(Ny):
            R0 = R[i, j]
            R1 = R[wrapx(i+1), j]
            R2 = R[wrapx(i+1), wrapy(j+1)]
            R3 = R[i, wrapy(j+1)]
            # Plaquette product (loop around in +kx,+ky orientation)
            loop = np.trace(R0 @ R1 @ R2 @ R3)
            phase = np.angle(loop)
            flux[i, j] = phase
            total_phase += phase

    Chern_cont = total_phase / (2*np.pi)
    return Chern_cont


def get_chern_number_from_single_particle_dm(single_particle_dm):
    """
    Calculate Chern number from single-particle density matrix with periodic boundary conditions.
    
    This is an alternative method to chern_from_mixed_spdm that uses derivatives.
    Uses the formula: C = (1/(2π)) * Im[Σ Tr(P @ (dP_dkx @ dP_dky - dP_dky @ dP_dkx))]
    
    Parameters
    ----------
    single_particle_dm : ndarray, shape (Nx, Ny, M, M)
        Single-particle density matrix on a momentum space grid
        
    Returns
    -------
    float
        Chern number
    """
    n_kx, n_ky = single_particle_dm.shape[0], single_particle_dm.shape[1]
    
    # Compute derivatives with wrapping for periodic boundary conditions
    # dP_dkx: difference along kx axis, wrapping from last to first
    dP_dkx = np.roll(single_particle_dm, -1, axis=0) - single_particle_dm
    
    # dP_dky: difference along ky axis, wrapping from last to first
    dP_dky = np.roll(single_particle_dm, -1, axis=1) - single_particle_dm
    
    # Compute integrand for all k-points
    integrand = np.zeros((n_kx, n_ky), dtype=complex)
    for i_kx, i_ky in product(range(n_kx), range(n_ky)):
        P = single_particle_dm[i_kx, i_ky, :, :]
        dP_dkx_ij = dP_dkx[i_kx, i_ky, :, :]
        dP_dky_ij = dP_dky[i_kx, i_ky, :, :]
        integrand[i_kx, i_ky] = np.trace(P @ (dP_dkx_ij @ dP_dky_ij - dP_dky_ij @ dP_dkx_ij))
    
    return (np.sum(integrand)/(2*np.pi)).imag


def qwx_projector_grid(Nx=31, Ny=31, m=-1.0):
    """
    Build P(k) for the Qi-Wu-Zhang model on an Nx x Ny grid.
    H(k) = sin kx σx + sin ky σy + (m + cos kx + cos ky) σz
    Occupied = lower band.
    """
    kx = np.linspace(-np.pi, np.pi, Nx, endpoint=False)
    ky = np.linspace(-np.pi, np.pi, Ny, endpoint=False)
    P = np.zeros((Nx, Ny, 2, 2), dtype=complex)
    for i, kxi in enumerate(kx):
        for j, kyj in enumerate(ky):
            hx = np.sin(kxi); hy = np.sin(kyj); hz = m + np.cos(kxi) + np.cos(kyj)
            H = np.array([[hz, hx - 1j*hy],
                          [hx + 1j*hy, -hz]], dtype=complex)
            w, V = np.linalg.eigh(H)
            # lower band = eigenvector with smaller eigenvalue
            idx = np.argsort(w)
            u = V[:, idx[0]]
            P[i, j] = np.outer(u, u.conj())
    return P

import numpy as np

def chern_from_spdm_with_threshold_eigenvalues(
    R,
    mode="projector",
    thresh=0.5,
    align="min",            # "min" -> trim to min rank over grid; "error" -> raise if ranks differ
    return_flux=False
):
    """
    Gauge-invariant Chern from a mixed SPDM R[kx,ky] (Nx,Ny,M,M), using eigenvalue thresholding.

    Parameters
    ----------
    R : ndarray, shape (Nx, Ny, M, M)
        Hermitian PSD single-particle density matrices.
    mode : {"projector", "amplitude"}
        "projector": take eigenvectors with eigval > thresh (then trim to constant rank).
        "amplitude": use V * sqrt(Lambda) for those eigenpairs; links are unitarized.
    thresh : float
        Eigenvalue threshold to pick the occupied subspace at each k.
    align : {"min","error"}
        How to enforce constant occupied dimension across k:
          - "min": use n_occ = min_k #(eig > thresh); trim each k to its top n_occ eigvals.
          - "error": raise if the counts differ.
    return_flux : bool
        If True, also return the plaquette phase grid (Nx,Ny).

    Returns
    -------
    C : float
        Chern number (continuous for mixed states).
    flux : ndarray, optional
        Plaquette phases in (-π, π], if return_flux=True.
    """
    assert R.ndim == 4 and R.shape[2] == R.shape[3], "R must be (Nx,Ny,M,M)"
    Nx, Ny, M, _ = R.shape

    # First pass: get counts above threshold to determine constant rank
    counts = np.zeros((Nx, Ny), dtype=int)
    evals_cache = [[None for _ in range(Ny)] for _ in range(Nx)]
    evecs_cache = [[None for _ in range(Ny)] for _ in range(Nx)]

    for ix in range(Nx):
        for iy in range(Ny):
            Rij = 0.5 * (R[ix, iy] + R[ix, iy].conj().T)  # hermitize
            w, V = np.linalg.eigh(Rij)                   # ascending
            evals_cache[ix][iy] = w
            evecs_cache[ix][iy] = V
            counts[ix, iy] = int(np.count_nonzero(w > thresh))

    n_occ = counts.min()
    if n_occ == 0:
        raise ValueError(
            f"No eigenvalues > {thresh} at some k; lower `thresh` or check R."
        )
    if (counts != n_occ).any():
        if align == "error":
            raise ValueError(
                f"Occupied dimension varies across k (min={n_occ}, max={counts.max()}). "
                f"Use align='min' or adjust `thresh`."
            )
        # else align == "min": silently trim to min count

    # Build frames of constant width n_occ at each k (sorted by descending eigenvalue)
    frames = [[None for _ in range(Ny)] for _ in range(Nx)]
    for ix in range(Nx):
        for iy in range(Ny):
            w = evals_cache[ix][iy]
            V = evecs_cache[ix][iy]
            # take indices of eigvals > thresh; if more than n_occ, keep the largest ones
            sel = np.where(w > thresh)[0]
            if sel.size > n_occ:
                sel = sel[np.argsort(w[sel])[::-1][:n_occ]]
            elif sel.size < n_occ:
                # If some point has exactly n_occ by definition of min(), this shouldn't happen
                # unless degeneracies touch the threshold. Take the top n_occ by value anyway.
                order = np.argsort(w)[::-1][:n_occ]
                sel = order
            # Order columns by descending eigenvalue for smoother links
            sel = sel[np.argsort(w[sel])[::-1]]
            if mode == "projector":
                F = V[:, sel]  # orthonormal frame (M, n_occ)
            elif mode == "amplitude":
                F = V[:, sel] @ np.diag(np.sqrt(np.clip(w[sel], 0.0, None)))
            else:
                raise ValueError("mode must be 'projector' or 'amplitude'")
            frames[ix][iy] = F

    def polar_unitary(X):
        """Unitary factor in the polar decomposition of X."""
        U, _, Vh = np.linalg.svd(X, full_matrices=False)
        return U @ Vh

    flux = np.zeros((Nx, Ny), dtype=float)
    total = 0.0

    for ix in range(Nx):
        ix1 = (ix + 1) % Nx
        for iy in range(Ny):
            iy1 = (iy + 1) % Ny

            F0 = frames[ix ][iy ]
            F1 = frames[ix1][iy ]
            F2 = frames[ix1][iy1]
            F3 = frames[ix ][iy1]

            # Non-Abelian links around the plaquette (oriented +kx, +ky, -kx, -ky)
            L01 = F0.conj().T @ F1
            L12 = F1.conj().T @ F2
            L23 = F2.conj().T @ F3
            L30 = F3.conj().T @ F0

            # Unitarize each link (robust even if frames are not perfectly orthonormal/matched)
            U01 = polar_unitary(L01)
            U12 = polar_unitary(L12)
            U23 = polar_unitary(L23)
            U30 = polar_unitary(L30)

            # Wilson loop (NO inverses)
            W = np.linalg.det(U01 @ U12 @ U23 @ U30)

            ph = np.angle(W)   # (-pi, pi]
            flux[ix, iy] = ph
            total += ph

    C = total / (2.0 * np.pi)
    return (C, flux) if return_flux else C



def chern_fhs_from_spdm(
    rho,
    thresh=0.5,
    return_diagnostics=False
):
    """
    Compute integer Chern number using the Fukui-Hatsugai-Suzuki (FHS) method.
    
    This method extracts the occupied subspace from a single-particle density matrix
    using eigenvalue thresholding, then computes the lattice Chern number via
    U(1) link variables and plaquette phases.
    
    The method is gauge-invariant and converges to the integer Chern number
    as the k-grid is refined, provided there is a clear gap between occupied
    and empty bands.
    
    Parameters
    ----------
    rho : ndarray, shape (Nx, Ny, M, M)
        Hermitian positive semidefinite single-particle density matrices
        on a 2D momentum grid. Eigenvalues should be in [0,1].
        Expected to have n_occ "full" bands (eigenvalues > thresh) and
        M - n_occ "empty" bands (eigenvalues < thresh) at each k-point.
    thresh : float, default=0.5
        Eigenvalue threshold to distinguish occupied from empty bands.
        Bands with eigenvalue > thresh are considered occupied.
    return_diagnostics : bool, default=False
        If True, return additional diagnostic information including:
        - plaquette flux grid
        - minimum occupation gap across all k-points
        - occupation counts
        
    Returns
    -------
    chern : float
        The computed Chern number (close to integer for clean systems).
    diagnostics : dict, optional
        If return_diagnostics=True, contains:
        - 'flux': ndarray (Nx, Ny), plaquette phases in (-π, π]
        - 'min_gap': float, minimum gap between lowest occupied and 
                     highest empty eigenvalue across all k
        - 'n_occ': int, number of occupied bands
        - 'gap_at_k': ndarray (Nx, Ny), occupation gap at each k-point
        
    Raises
    ------
    AssertionError
        If the number of occupied bands differs across k-points.
        
    Notes
    -----
    The FHS method works as follows:
    1. At each k, diagonalize ρ(k) and select eigenvectors with eigenvalue > thresh
       as the occupied frame Ψ(k) of shape (M, n_occ).
    2. Compute overlap matrices M_x(k) = Ψ†(k) Ψ(k+x̂), M_y(k) = Ψ†(k) Ψ(k+ŷ)
    3. Extract U(1) link variables: U_x(k) = det(M̃_x)/|det(M̃_x)| where M̃ is the
       unitary part of M (from polar decomposition).
    4. Compute plaquette flux: F(k) = Arg[U_x(k) U_y(k+x̂) U_x(k+ŷ)^-1 U_y(k)^-1]
    5. Sum: C = (1/2π) Σ_k F(k)
    
    References
    ----------
    Fukui, Hatsugai, Suzuki, J. Phys. Soc. Jpn. 74, 1674 (2005)
    """
    assert rho.ndim == 4 and rho.shape[2] == rho.shape[3], \
        f"rho must be shape (Nx, Ny, M, M), got {rho.shape}"
    Nx, Ny, M, _ = rho.shape
    
    # First pass: diagonalize all k-points and count occupied bands
    evals_grid = np.zeros((Nx, Ny, M), dtype=float)
    evecs_grid = np.zeros((Nx, Ny, M, M), dtype=complex)
    n_occ_grid = np.zeros((Nx, Ny), dtype=int)
    
    for ix in range(Nx):
        for iy in range(Ny):
            # Hermitize for numerical stability
            rho_k = 0.5 * (rho[ix, iy] + rho[ix, iy].conj().T)
            w, V = np.linalg.eigh(rho_k)  # eigenvalues in ascending order
            # Store in descending order (largest eigenvalue first)
            evals_grid[ix, iy] = w[::-1]
            evecs_grid[ix, iy] = V[:, ::-1]
            n_occ_grid[ix, iy] = np.count_nonzero(w > thresh)
    
    # Assert uniform occupied dimension across all k-points
    n_occ = n_occ_grid[0, 0]
    assert np.all(n_occ_grid == n_occ), \
        f"Number of occupied bands must be the same at all k-points. " \
        f"Found counts ranging from {n_occ_grid.min()} to {n_occ_grid.max()}. " \
        f"Adjust threshold (currently {thresh}) or check your density matrix."
    
    assert n_occ > 0, \
        f"No eigenvalues > {thresh} found. Lower threshold or check density matrix."
    assert n_occ < M, \
        f"All eigenvalues > {thresh}. Raise threshold or check density matrix."
    
    # Compute occupation gap at each k-point: gap = (lowest occupied) - (highest empty)
    # evals_grid is sorted descending, so occupied are indices 0:n_occ, empty are n_occ:M
    gap_at_k = evals_grid[:, :, n_occ - 1] - evals_grid[:, :, n_occ]
    min_gap = gap_at_k.min()
    
    # Build occupied frames: Ψ(k) is (M, n_occ) matrix of occupied eigenvectors
    # Already sorted by descending eigenvalue
    frames = evecs_grid[:, :, :, :n_occ]  # shape (Nx, Ny, M, n_occ)
    
    def polar_unitary(X):
        """Extract unitary factor from polar decomposition via SVD."""
        U, _, Vh = np.linalg.svd(X, full_matrices=False)
        return U @ Vh
    
    def u1_link(M_overlap):
        """
        Compute U(1) link variable from overlap matrix.
        U = det(M̃) / |det(M̃)| where M̃ is the unitary part of M.
        """
        M_unitary = polar_unitary(M_overlap)
        d = np.linalg.det(M_unitary)
        # Normalize to unit circle (handles numerical errors)
        return d / np.abs(d) if np.abs(d) > 1e-12 else 1.0
    
    # Compute U(1) link variables in x and y directions
    link_x = np.zeros((Nx, Ny), dtype=complex)
    link_y = np.zeros((Nx, Ny), dtype=complex)
    
    for ix in range(Nx):
        ix_next = (ix + 1) % Nx
        for iy in range(Ny):
            iy_next = (iy + 1) % Ny
            
            Psi_k = frames[ix, iy]                    # (M, n_occ)
            Psi_kx = frames[ix_next, iy]              # (M, n_occ)
            Psi_ky = frames[ix, iy_next]              # (M, n_occ)
            
            # Overlap matrices (n_occ x n_occ)
            M_x = Psi_k.conj().T @ Psi_kx
            M_y = Psi_k.conj().T @ Psi_ky
            
            link_x[ix, iy] = u1_link(M_x)
            link_y[ix, iy] = u1_link(M_y)
    
    # Compute plaquette flux: F(k) = Arg[U_x(k) * U_y(k+x̂) * U_x(k+ŷ)^-1 * U_y(k)^-1]
    # Since U_x, U_y are on the unit circle, U^-1 = U.conj()
    flux = np.zeros((Nx, Ny), dtype=float)
    total_flux = 0.0
    
    for ix in range(Nx):
        ix_next = (ix + 1) % Nx
        for iy in range(Ny):
            iy_next = (iy + 1) % Ny
            
            # Plaquette product going counterclockwise: 
            # k -> k+x̂ -> k+x̂+ŷ -> k+ŷ -> k
            plaquette = (
                link_x[ix, iy] *                    # k -> k+x̂
                link_y[ix_next, iy] *               # k+x̂ -> k+x̂+ŷ  
                link_x[ix, iy_next].conj() *        # k+ŷ -> k+x̂+ŷ (reversed, so conjugate)
                link_y[ix, iy].conj()               # k -> k+ŷ (reversed, so conjugate)
            )
            
            phase = np.angle(plaquette)  # in (-π, π]
            flux[ix, iy] = phase
            total_flux += phase
    
    chern = total_flux / (2.0 * np.pi)
    
    if return_diagnostics:
        diagnostics = {
            'flux': flux,
            'min_gap': min_gap,
            'n_occ': n_occ,
            'gap_at_k': gap_at_k,
        }
        return chern, diagnostics
    
    return chern


if __name__ == "__main__":
    # Test with Qi-Wu-Zhang model (should give Chern = -1 for m = -1)
    single_particle_dm = qwx_projector_grid(41, 41, m=-1.0)
    
    print("Testing different Chern number methods on QWZ model (m=-1, expect C=-1):")
    print("-" * 60)
    
    C1 = chern_from_mixed_spdm(single_particle_dm)
    print(f"chern_from_mixed_spdm:                    C = {C1:.6f}")
    
    C2 = get_chern_number_from_single_particle_dm(single_particle_dm)
    print(f"get_chern_number_from_single_particle_dm: C = {C2:.6f}")
    
    C3 = chern_from_spdm_with_threshold_eigenvalues(single_particle_dm)
    print(f"chern_from_spdm_with_threshold_eigenvalues: C = {C3:.6f}")
    
    C4, diag = chern_fhs_from_spdm(single_particle_dm, return_diagnostics=True)
    print(f"chern_fhs_from_spdm (FHS method):         C = {C4:.6f}")
    print(f"  -> n_occ = {diag['n_occ']}, min_gap = {diag['min_gap']:.4f}")