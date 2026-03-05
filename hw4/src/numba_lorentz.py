import numpy as np
from numba import njit, prange
from numba.core import cgutils
import numpy as np

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
    counts = np.zeros(bins) # Initialize counts
    for i in prange(n):
        u = np.random.random() # Uniform(0,1)
        x = 1. / np.tan(np.pi * u) # x = 1/tan(pi*u)
        ix = int((x - xmin) * xfac) # Map x to bin index
        if 0 <= ix < bins:
            cgutils.atomic_add(counts, ix, 1) # Atomic increment
    return counts
