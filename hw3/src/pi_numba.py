import random, sys
import timeit
from numba import jit, njit, prange

@njit
def calc_pi_numba(n: float) -> float:
    h = 0
    for _ in range(n):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        if x*x + y*y < 1.:
            h += 1
    return 4. * h / n

@jit(nopython=True, nogil=True, parallel=True)
def calc_pi_parallel(n: float) -> float:
    h = 0
    for _ in prange(n):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        if x**2 + y**2 < 1:
            h += 1
    return 4. * h / n

if __name__ == "__main__":
    n_value = float(sys.argv[1])
    if n_value <= 0 or not n_value.is_integer():
        raise ValueError("n must be a positive integer (e.g. 10000000 or 1e7)")
    n = int(n_value)
    #pi_est = calc_pi_numba(n)
    pi_est = calc_pi_parallel(n)
    print(f"n={n}, pi={pi_est}")