from calc_pi import calc_pi_cython
from time import time
logn = 8 # Number of samples
n = 10**logn # Number of samples
start = time() # Start timer
pi = calc_pi_cython(n) # Calculate pi
end = time() # End timer
print(f"Estimated Pi: {pi} [Samples: 10^{logn}, Time: {end - start:.3f} s]")