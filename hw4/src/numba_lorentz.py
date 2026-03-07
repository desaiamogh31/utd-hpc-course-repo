import numpy as np
from numba import njit, prange, get_num_threads

@njit(parallel=True, nogil=True)
def lorentzian_histogram_numba(n, bins=100, xmin=-10, xmax=10):
    """
    Sample n random points from the Lorentzian distribution
    using inverse transform sampling. Make a histogram with
    the specified bin count and range. Returns counts.
    """
    n = int(n) # Ensure n is an integer
    np.random.seed(42)  # Set global seed for reproducibility
    xfac = bins / (xmax - xmin) # Factor to map x to bin index
    
    # Allocate per-thread histogram storage (each thread gets its own row)
    num_threads = get_num_threads()
    local_histograms = np.zeros((num_threads, bins))
    
    for i in prange(n):
        u = np.random.random() # Uniform(0,1)
        x = 1. / np.tan(np.pi * u) # x = 1/tan(pi*u)
        ix = int((x - xmin) * xfac) # Map x to bin index
        if 0 <= ix < bins:
            # Assign to thread based on iteration modulo num_threads
            thread_id = i % num_threads
            local_histograms[thread_id, ix] += 1
    
    # Aggregate all thread-local histograms
    counts = np.zeros(bins)
    for thread_id in range(num_threads):
        for j in range(bins):
            counts[j] += local_histograms[thread_id, j]
    return counts
