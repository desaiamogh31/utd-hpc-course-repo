import numpy as np
import threading
from lorentzian import lorentzian_histogram
def add_chunk(n, counts, lock, bins=100, xmin=-10, xmax=10):
    """
    Generate n samples and add to global counts.
    """
    local_counts = lorentzian_histogram(n, bins, xmin, xmax)
    # Acquire lock to merge partial counts into global
    with lock:
        counts += local_counts


def run_threaded(n, n_threads=4, bins=100, xmin=-10, xmax=10):
    """
    Run the Lorentzian sampling in parallel using threads.
    """
    n=int(n)
    # Split n samples among processes
    chunks = (n // n_threads) * np.ones(n_threads, dtype=int) #Number of samples per thread
    chunks[:n % n_threads] += 1 # Distribute remainder, if remainder is x then the first x threads will get 1 extra sample each
    threads = [None] * n_threads # Thread list
    counts = np.zeros(bins) # Global counts
    lock = threading.Lock() # Lock for global data
    cnt=0
    for i in range(n_threads):
        t = threading.Thread(target=add_chunk, args=(chunks[i], counts, lock, bins, xmin, xmax))
        t.start() # Start thread
        threads[i] = t
        cnt+=1
    for t in threads:
        t.join() # Wait for all threads to finish
    return counts,cnt

