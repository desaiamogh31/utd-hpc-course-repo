from numpy import sum
from numpy.random import rand
import sys
def calc_pi_numpy(n:float):
    h = sum(rand(n)**2 + rand(n)**2 < 1.)
    return 4. * float(h) / float(n) # Estimate pi
if __name__ == "__main__":
    n = int(float(sys.argv[1]))
    pi_est = calc_pi_numpy(n)
    print(f"n={n}, pi={pi_est}")