from mpire import WorkerPool
import numpy as np


def _lorentzian_histogram_seeded(n, bins, xmin, xmax, seed_int):
    """
    Worker-local Lorentzian histogram using an independent RNG seed.
    """
    n = int(n)
    rng = np.random.default_rng(seed_int)
    u = rng.random(int(n))
    x = 1. / np.tan(np.pi * u)  # Lorentzian inverse CDF: cot(pi*u)
    counts, _ = np.histogram(x, bins=bins, range=(xmin, xmax))
    return counts


def run_mpire(n, n_jobs=4, bins=100, xmin=-10, xmax=10, base_seed=42):
    """
    Run the Lorentzian sampling in parallel using mpire with independent
    deterministic RNG streams per worker.
    """
    # Split n samples among jobs
    n = int(n)
    chunks = (n // n_jobs) * np.ones(n_jobs, dtype=int)
    chunks[:n % n_jobs] += 1 # Distribute remainder
    seed_seq = np.random.SeedSequence(base_seed)
    child_seeds = [int(child.entropy) for child in seed_seq.spawn(n_jobs)]
    work_items = [
        (int(chunks[i]), bins, xmin, xmax, child_seeds[i])
        for i in range(n_jobs)
    ]

    with WorkerPool(n_jobs=n_jobs) as pool:
        results = pool.map(_lorentzian_histogram_seeded, work_items, concatenate_numpy_output=False)
    return np.sum(results, axis=0)
