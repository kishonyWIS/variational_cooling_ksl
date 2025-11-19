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

# Example:
# P = qwx_projector_grid(41, 41, m=-1.0)
# C, Craw, F = chern_fhs_from_projector(P, purify=False, return_debug=True)
# print(C, Craw, _.get('warnings', {}))


single_particle_dm = qwx_projector_grid(41, 41, m=-1.0)
C = chern_from_mixed_spdm(
    single_particle_dm)
print("Chern =", C)