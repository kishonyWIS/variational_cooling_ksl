import numpy as np

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