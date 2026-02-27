import numpy as np
import multiprocessing


def lorentzian_histogram(n, bins=100, xmin=-10, xmax=10):
    """
    Sample n random points from the Lorentzian distribution
    using inverse transform sampling. Make a histogram with
    the specified bin count and range. Returns counts.
    """
    n=int(n)
    u = np.random.random(n) # Uniform(0,1)
    x = 1. / np.tan(np.pi * u) # x = 1/tan(pi*u)
    counts, _ = np.histogram(x, bins=bins, range=(xmin, xmax))
    return counts # No need to return bin edges for uniform bins



def run_multiproc(n, n_cores=4, bins=100, xmin=-10, xmax=10):
    """
    Run the Lorentzian sampling in parallel using processes.
    """
    # Split n samples among processes
    n=int(n)
    chunks = (n // n_cores) * np.ones(n_cores, dtype=int)
    chunks[:n % n_cores] += 1 # Distribute remainder
    # Use partial function to reset default arguments (bins, xmin, xmax)
    from functools import partial
    lorentzian_hist_func = partial(lorentzian_histogram, bins=bins, xmin=xmin, xmax=xmax)
    # Use Pool to distribute chunks to processes
    with multiprocessing.Pool(n_cores) as pool: #multiprocessing.Pool(...) starts a pool with n_cores separate python worker processes
        results = pool.map(lorentzian_hist_func, chunks)
    return np.sum(results, axis=0) # Aggregate results