import numpy as np
from concurrent.futures import ProcessPoolExecutor
from lorentzian import lorentzian_histogram


def run_ppe(n, max_workers=4, bins=100, xmin=-10, xmax=10, base_seed=42):
    """
    Run the Lorentzian sampling in parallel using ProcessPoolExecutor.
    """
    n = int(n) # Ensure n is an integer
    chunks = (n // max_workers) * np.ones(max_workers, dtype=int) # Split n samples among workers
    chunks[:n % max_workers] += 1 # Distribute remainder
    # Create deterministic per-worker seeds (one per worker)
    seeds = [int(base_seed + i) for i in range(max_workers)]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(lorentzian_histogram, int(chunks[i]), bins, xmin, xmax, seeds[i]) for i in range(max_workers)]
        results = [f.result() for f in futures] # Collect results
    return np.sum(results, axis=0) # Aggregate results