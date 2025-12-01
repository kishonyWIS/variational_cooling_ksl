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



if __name__ == "__main__":
    single_particle_dm = qwx_projector_grid(41, 41, m=-1.0)
    C = chern_from_mixed_spdm(
        single_particle_dm)
    print("Chern =", C)