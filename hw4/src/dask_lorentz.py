import dask
from dask import delayed
import numpy as np
from lorentzian import lorentzian_histogram

@delayed
def delayed_lorentzian_histogram(n, bins=100, xmin=-10, xmax=10):
    """
    Delayed function for lorentzian_histogram.
    """
    return lorentzian_histogram(n, bins, xmin, xmax)

def run_dask(n:float, n_tasks=4):
    """
    Run the Lorentzian sampling in parallel using Dask.
    """
    n=int(n) # Ensure n is an integer
    # Split n samples among tasks
    chunks = (n // n_tasks) * np.ones(n_tasks, dtype=int)
    chunks[:n % n_tasks] += 1 # Distribute remainder
    tasks = [delayed_lorentzian_histogram(chunk) for chunk in chunks]
    results = dask.compute(*tasks) # Compute all tasks
    print(f"Results shapes: {np.shape(results)}") # Debug print
    return np.sum(results, axis=0) # Aggregate results