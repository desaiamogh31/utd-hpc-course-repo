import numpy as np
import multiprocessing
from lorentzian import lorentzian_histogram



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