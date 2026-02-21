import numpy as np
from scipy.linalg import eigh
import argparse
import matplotlib.pyplot as plt

VALID_POTENTIALS = ['well', 'harmonic', 'anisotropic_harmonic']
def build_2d_hamiltonian_sparse(N=20, potential="well"):
    """
    Build a sparse (CSR) discretized 2D Hamiltonian on an N x N grid.

    H approximates:  -∂^2/∂x^2 - ∂^2/∂y^2 + V(x,y)
    with a 5-point stencil.
    """
    dx = 1.0 / float(N)
    inv_dx2 = 1.0 / (dx * dx)  # = N^2

    n = N * N
    H = lil_matrix((n, n), dtype=np.float64)

    def idx(i, j):
        return i * N + j


    def V(i, j):
        #Boundary conditions: Virtual Infinite potential outside the box (Dirichlet BCs)
        if i == 0 or i == N - 1 or j == 0 or j == N - 1:
            return float(N**2)
        if potential == "well":
            return 0.0
        elif potential == "harmonic":
            x = (i - N / 2.0) * dx
            y = (j - N / 2.0) * dx
            return 4.0 * (x * x + y * y)
        # Example 3: Anisotropic harmonic oscillator; stronger confinement in x
        elif potential == 'anisotropic_harmonic':
            x = (i - N/2) * dx
            y = (j - N/2) * dx
            return 4. * (x**2 + 0.2*y**2)
        else:
            raise ValueError(f"Unknown potential: {potential}")
    # Build the matrix: For each (i, j), set diagonal for 2D Laplacian plus V
    for i in range(N):
        for j in range(N):
            row = idx(i, j)

            # Kinetic term: (-∇^2) with 5-point stencil
            # diag = 4/dx^2, neighbors = -1/dx^2
            H[row, row] = 4.0 * inv_dx2 + V(i, j)

            if i > 0:
                H[row, idx(i - 1, j)] = -inv_dx2
            if i < N - 1:
                H[row, idx(i + 1, j)] = -inv_dx2
            if j > 0:
                H[row, idx(i, j - 1)] = -inv_dx2
            if j < N - 1:
                H[row, idx(i, j + 1)] = -inv_dx2

    return H.tocsr()  # eigsh works best with CSR/CSC

def solve_eigen_sparse(N=20, potential="well", n_eigs=5):
    """
    Compute the lowest n_eigs eigenvalues/eigenvectors using sparse eigensolver.
    """

    H = build_2d_hamiltonian_sparse(N, potential)

    # which='SA' => smallest algebraic eigenvalues (what you want for ground state)
    vals, vecs = eigsh(H, k=n_eigs, which="SA")

    # eigsh does NOT guarantee sorted output
    idx_sort = np.argsort(vals)
    vals = vals[idx_sort]
    vecs = vecs[:, idx_sort]
    return vals, vecs

if __name__ == '__main__':
    # Example local test

    parser = argparse.ArgumentParser(description="Solve 2D Hamiltonian eigenvalue problem.")
    parser.add_argument('--N', type=int, required=True)
    parser.add_argument('--potential', choices=VALID_POTENTIALS, required=True)
    parser.add_argument('--n-eigs', type=int, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--density-out", type=str, default=None,
                        help="Optional output file for ground-state probability density |psi(x,y)|^2.")
    parser.add_argument("--density-plot", type=str, default=None,
                        help="Optional image file for imshow plot of ground-state density. If omitted, defaults to <density-out>.png.")
    args = parser.parse_args()
    if args.N <= 0:
        raise ValueError("N must be a positive integer.")
    if args.n_eigs <= 0:
        raise ValueError("n_eigs must be a positive integer.")
    if args.n_eigs > args.N * args.N:
        raise ValueError("n_eigs cannot exceed N^2.")
    vals, vecs = solve_eigen(N=args.N, potential=args.potential, n_eigs=args.n_eigs)
    print(f"Lowest {args.n_eigs} eigenvalues:", vals)
    np.savetxt(args.out, vals)

    # Optionally save the ground-state probability density and plot
    if args.density_out is not None:
        psi0 = vecs[:, 0].reshape((args.N, args.N)) # Ground-state wavefunction, flatten
        prob_density = np.abs(psi0) ** 2
        np.savetxt(args.density_out, prob_density)

        plot_path = args.density_plot if args.density_plot is not None else f"{args.density_out}.png"
        plt.figure()
        plt.imshow(prob_density, origin='lower', extent=[1, args.N, 1, args.N], aspect='auto')
        plt.colorbar(label=r'$|\psi(x,y)|^2$')
        plt.title('Ground-state Probability Density')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"Saved ground-state density to {args.density_out}")
        print(f"Saved ground-state density plot to {plot_path}")