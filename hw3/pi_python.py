from numpy.random import rand
import sys
def calc_pi_loop(n):
    h = 0 # Number of hits inside the circle
    for _ in range(n):
        x, y = rand(), rand() # Random points in [0, 1)
        if x*x + y*y < 1.:
            h += 1 # Successful hit
    return 4. * float(h) / float(n) # Estimate pi
if __name__ == "__main__":
    #argv = ["pi_python.py", "1000000"] # Example command-line arguments
    #argv[0] = "pi_python.py" # Script name
    #argv[1] = "1000000" # Number of random points
    #Can include a fail safe for when the user does not provide a command-line argument for n
    # if len(sys.argv) == 1:
    #     print("No command-line argument provided for n. Using default value of 1000000.")
    #     n = 1000000
    # elif len(sys.argv) == 2:
    #     n = int(sys.argv[1])
    # else:
    #     print("Invalid number of command-line arguments.")
    #     sys.exit(1)
    n = int(sys.argv[1]) # Command-line argument
    pi_est = calc_pi_loop(n)
    print(f"n={n}, pi={pi_est}")