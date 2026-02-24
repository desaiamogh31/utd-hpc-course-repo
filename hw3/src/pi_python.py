from numpy.random import rand
import sys
def calc_pi_loop(n:float) -> float:
    n = int(n) # Ensure n is an integer
    h = 0 # Number of hits inside the circle
    for _ in range(n):
        x, y = rand(), rand() # Random points in [0, 1)
        if x*x + y*y < 1.:
            h += 1 # Successful hit
    return 4. * float(h) / float(n) # Estimate pi
if __name__ == "__main__":
    n = int(sys.argv[1]) # Command-line argument
    pi_est = calc_pi_loop(n)
    print(f"n={n}, pi={pi_est}")