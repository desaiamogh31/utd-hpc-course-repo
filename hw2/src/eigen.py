import numpy as np
from scipy.linalg import eigh
import argparse
import matplotlib.pyplot as plt

VALID_POTENTIALS = ['well', 'harmonic', 'anisotropic harmonic']
def build_2d_hamiltonian(N=20, potential='well'):
    """
    Build a discretized 2D Hamiltonian on an N x N grid.
    Parameters
    ----------
    N : int
    potential : str
    Number of points in each dimension (N^2 total points).
    Choose the potential. 'well' or 'harmonic' examples.
    Returns
    -------
    H : ndarray of shape (N^2, N^2)
    The Hamiltonian matrix approximating -d^2/dx^2 - d^2/dy^2 + V(x,y).
    """
    dx = 1. / float(N) # grid spacing
    inv_dx2 = float(N * N) # 1/dx^2
    H = np.zeros((N*N, N*N), dtype=np.float64)
    # Helper function to map (i,j) -> linear index
    def idx(i, j):
        return i * N + j
        # Potential function
    def V(i, j):
        # Example 1: infinite square well -> zero in interior, large outside
        if potential == 'well':
            # No boundary enforcement here, but can skip boundary wavefunction
            return 0.
        # Example 2: 2D harmonic oscillator around center
        elif potential == 'harmonic':
            x = (i - N/2) * dx
            y = (j - N/2) * dx
            # Quadratic potential V = k * (x^2 + y^2)
            return 4. * (x**2 + y**2)
        # Example 3: Anisotropic harmonic oscillator; stronger confinement in x
        elif potential == 'anisotropic harmonic':
            x = (i - N/2) * dx
            y = (j - N/2) * dx
            return 4. * (x**2 + 0.2*y**2)
        else:
            return 0.
    # Build the matrix: For each (i, j), set diagonal for 2D Laplacian plus V
    for i in range(N):
        for j in range(N):
            row = idx(i,j)
            # Potential
            H[row, row] = +4. * inv_dx2 + V(i,j) # "Kinetic" ~ +4/dx^2 in 2D FD
            # Neighbors (assuming no boundary conditions or Dirichlet)
            if i > 0: # up
                H[row, idx(i-1, j)] = inv_dx2
            if i < N-1: # down
                H[row, idx(i+1, j)] = inv_dx2
            if j > 0: # left
                H[row, idx(i, j-1)] = inv_dx2
            if j < N-1: # right
                H[row, idx(i, j+1)] = inv_dx2
    return H
def solve_eigen(N=20, potential='well', n_eigs=None):   
    H = build_2d_hamiltonian(N, potential)
    # Solve entire spectrum (careful for large N)
    vals, vecs = eigh(H)
    # Sort
    idx_sorted = np.argsort(vals)
    vals_sorted = vals[idx_sorted]
    vecs_sorted = vecs[:, idx_sorted]
    
    if n_eigs is None:
        return vals_sorted, vecs_sorted
    else:
        return vals_sorted[:n_eigs], vecs_sorted[:, :n_eigs]

if __name__ == '__main__':
    # Example local test

    parser = argparse.ArgumentParser(description="Solve 2D Hamiltonian eigenvalue problem.")
    parser.add_argument('--N', type=int, required=True)
    parser.add_argument('--potential', choices=VALID_POTENTIALS, required=True)
    parser.add_argument('--n-eigs', type=int, required=True)
    #parser.add_argument("--out", type=str, required=True) 
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
    #np.savetxt(args.out, vals)

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