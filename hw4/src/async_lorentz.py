import numpy as np
import asyncio
from lorentzian import lorentzian_histogram

async def async_lorentzian_histogram(n, bins=100, xmin=-10, xmax=10):
    """
    Async wrapper that offloads the CPU-bound histogram generation
    to a thread so the asyncio event loop isn't blocked.
    """
    return await asyncio.to_thread(lorentzian_histogram, n, bins, xmin, xmax) #await 

async def add_chunk(n, bins=100, xmin=-10, xmax=10, n_subchunks=10):
    """
    Generate n samples in subchunks and return the summed counts for
    this chunk. Does not mutate shared arrays so it works with
    thread/process offloading cleanly.
    """
    # Split n samples among sub-chunks
    sub_chunks = (n // n_subchunks) * np.ones(n_subchunks, dtype=int)
    sub_chunks[:n % n_subchunks] += 1
    # Offload each subchunk to a thread so work can run in parallel
    local_counts = await asyncio.gather(*[
        asyncio.to_thread(lorentzian_histogram, int(chunk), bins, xmin, xmax)
        for chunk in sub_chunks
    ])
    return np.sum(local_counts, axis=0)

async def get_counts(n, n_tasks=4, bins=100, xmin=-10, xmax=10, n_subchunks=10):
    """
    Async function to run the Lorentzian sampling in parallel using asyncio.
    Each task returns its own counts array and we sum them at the end.
    """
    # Split n samples among tasks
    chunks = (n // n_tasks) * np.ones(n_tasks, dtype=int)
    chunks[:n % n_tasks] += 1
    tasks = [
        asyncio.create_task(add_chunk(int(chunk), bins, xmin, xmax, n_subchunks))
        for chunk in chunks
    ]
    results = await asyncio.gather(*tasks)
    counts = np.sum(results, axis=0)
    return counts

def run_async(n, n_tasks=4, bins=100, xmin=-10, xmax=10, n_subchunks=10):
    """
    Run the Lorentzian sampling in parallel using asyncio.
    """
    n = int(n)
    return asyncio.run(get_counts(n, n_tasks, bins, xmin, xmax, n_subchunks))

if __name__ == "__main__":
    import time
    #Strong Scaling
    n=1e7
    times=[]
    thread_counts = [1, 2, 4]
    for n_threads in thread_counts:
        start_time = time.perf_counter()
        counts = run_async(n*n_threads, n_tasks=n_threads, bins=100, xmin=-10, xmax=10, n_subchunks=10)
        end_time = time.perf_counter()
        thread_time =  end_time - start_time
        times.append(thread_time)
        print(f"Threads: {n_threads}, Time: {thread_time:.3f} seconds")
    
    np.savetxt("async_lorentzian_strong_scaling_times.txt",
        np.column_stack([thread_counts, times]),
        fmt="%d %.6f")
    
    #Weak Scaling
    n=1e6
    times=[]
    thread_counts = [1, 2, 4] # Adjust thread counts as needed
    for n_threads in thread_counts:
        start_time = time.perf_counter()
        counts = run_async(n*n_threads, n_tasks=n_threads, bins=100, xmin=-10, xmax=10, n_subchunks=10)
        end_time = time.perf_counter()
        thread_time =  end_time - start_time
        times.append(thread_time)
        print(f"Threads: {n_threads}, Time: {thread_time:.3f} seconds")
    
    np.savetxt("async_lorentzian_weak_scaling_times.txt",
        np.column_stack([thread_counts, times]),
        fmt="%d %.6f")
    


