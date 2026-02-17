import random, sys
from numba import jit, njit, prange

@njit
def calc_pi_numba(n):
    h = 0
    for _ in range(n):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        if x*x + y*y < 1.:
            h += 1
    return 4. * h / n

@jit(nopython=True, nogil=True, parallel=True)
def calc_pi_parallel(n):
    h = 0
    for _ in prange(n):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        if x**2 + y**2 < 1:
            h += 1
    return 4. * h / n

if __name__ == "__main__":
    n = int(sys.argv[1])
    #pi_est = calc_pi_numba(n)
    pi_est = calc_pi_parallel(n)
    print(f"n={n}, pi={pi_est}")