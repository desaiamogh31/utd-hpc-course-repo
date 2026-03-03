from mpire import WorkerPool
import numpy as np

def lorentzian_histogram(n, bins=100, xmin=-10, xmax=10):
    """
    Sample n random points from the Lorentzian distribution
    using inverse transform sampling. Make a histogram with
    the specified bin count and range. Returns counts.
    """
    n = int(n) # Ensure n is an integer
    np.random.seed(42)  # Set global seed for reproducibility
    u = np.random.random(n) # Uniform(0,1)
    x = 1. / np.tan(np.pi * u) # x = 1/tan(pi*u)
    counts, _ = np.histogram(x, bins=bins, range=(xmin, xmax))
    return counts # No need to return bin edges for uniform bins

def run_mpire(n, n_jobs=4, bins=100, xmin=-10, xmax=10):
    """
    Run the Lorentzian sampling in parallel using mpire.
    """
    # Split n samples among jobs
    chunks = (n // n_jobs) * np.ones(n_jobs, dtype=int)
    chunks[:n % n_jobs] += 1 # Distribute remainder
    with WorkerPool(n_jobs=n_jobs) as pool:
        # See mpire docs for argument passing; alternatively use starmap
        results = pool.map(lorentzian_histogram, chunks)
    return np.sum(results, axis=0) # Aggregate results