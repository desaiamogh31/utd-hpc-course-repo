print("Hello World from Python!")
import argparse

# parser = argparse.ArgumentParser()

# parser.add_argument('--name', required=True)
# parser.add_argument('--age', type=int, required=True)

# args = parser.parse_args()

# print("Hello", args.name)
# print("You are", args.age, "years old.")

# import argparse

VALID_POTENTIALS = ['well', 'harmonic']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve 2D Hamiltonian eigenvalue problem.")
    parser.add_argument('--N', type=int, required=True)
    parser.add_argument('--potential', choices=VALID_POTENTIALS, required=True)
    parser.add_argument('--n-eigs', type=int, required=True)

    args = parser.parse_args()

    if args.N <= 0:
        raise ValueError("N must be positive.")
    if args.n_eigs <= 0:
        raise ValueError("n-eigs must be positive.")
    if args.n_eigs > args.N * args.N:
        raise ValueError("n-eigs cannot exceed N^2.")

    #vals, vecs = solve_eigen(N=args.N, potential=args.potential, n_eigs=args.n_eigs)

    print("Code successfully ran with N =", args.N, "potential =", args.potential, "n-eigs =", args.n_eigs)