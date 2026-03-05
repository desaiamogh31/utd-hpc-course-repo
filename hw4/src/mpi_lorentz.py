import numpy as np
from mpi4py import MPI

def lorentzian_histogram(n, bins=100, xmin=-10, xmax=10, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    u = rng.random(n)
    x = 1. / np.tan(np.pi * u)
    counts, _ = np.histogram(x, bins=bins, range=(xmin, xmax))
    return counts.astype(np.int64)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Start timing
if rank == 0:
    import time
    start_time = time.time()

# Independent RNG stream per rank
seed = 42
ss = np.random.SeedSequence(seed)
child = ss.spawn(size)[rank]
rng = np.random.default_rng(child)

n_total = 10_000_000
chunks = np.full(size, n_total // size, dtype=int)
chunks[: n_total % size] += 1
local = lorentzian_histogram(int(chunks[rank]), bins=100, xmin=-10, xmax=10, rng=rng)
global_counts = np.empty_like(local)
comm.Allreduce(local, global_counts, op=MPI.SUM)

if rank == 0:
    import time
    end_time = time.time()
    print(f"Total samples: {n_total}")
    print(f"Runtime: {end_time - start_time:.3f} seconds")
    print(f"Samples per second: {n_total / (end_time - start_time):.0f}")
    
    # Save results
    bin_edges = np.linspace(-10, 10, 101)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    np.savetxt("lorentzian_histogram.txt",
        np.column_stack([bin_centers, global_counts]),
        fmt="%.6f %d")
    print("Results saved to lorentzian_histogram.txt")