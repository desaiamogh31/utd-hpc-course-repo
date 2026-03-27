import numpy as np

def a(x, G, m, epsilon=0.01):
    """The acceleration of all masses."""
    assert epsilon > 0, "Epsilon must be greater than zero"
    dx = x - x[:,np.newaxis]  # Difference in position
    dx2 = np.sum(dx**2, axis=2)  # Distance squared
    dx2_softened = dx2 + epsilon**2  # Softened distance squared
    dx3 = dx2_softened**1.5  # Softened distance cubed
    # a_i = G * sum_{j≠i} m_j * (x_j - x_i) / |x_j - x_i|^3
    return G * np.sum(dx * (m / dx3)[:,:,np.newaxis], axis=1)

def timestep(x0, v0, G, m, dt, epsilon=0.01):
    """Computes the next position and velocity for all masses
    given initial conditions and a time step size. """
    a0 = a(x0, G, m, epsilon)  # Initial acceleration
    v1 = a0 * dt + v0  # New velocity
    x1 = v1 * dt + x0  # New position
    return x1, v1

def initial_conditions(N, D, x_range=(0, 1), v_range=(0, 0), m_value=1.):
    """Generates initial conditions for N uniform masses with random
    starting positions and velocities in D-dimensional space."""
    np.random.seed(0)  # Set random seed for reproducibility
    x0 = np.random.uniform(*x_range, size=(N, D))  # Random initial positions
    v0 = np.random.uniform(*v_range, size=(N, D))  # Random initial velocities
    m = np.full(N, m_value, dtype=np.float64)  # Uniform masses
    return x0, v0, m

def simulate(N=100, D=3, G=0.5, m=1., dt=1e-3, t_max=1., T=None, epsilon=0.01, x_range=(0, 1), v_range=(0, 0)):
    """Simulates the motion of N masses in D-dimensional space
    under the influence of gravity for a given time period."""
    x0, v0, m = initial_conditions(N, D, x_range, v_range, m)  # Initial conditions
    if T is None:  # If T is not given
        T = int(t_max / dt)  # Number of time steps
        dt = t_max / float(T)  # Adjusted time step size
    else:
        T = int(T)  # Ensure T is an integer
        t_max = float(T) * dt  # Adjusted maximum time
    x = np.zeros([T+1, N, D])  # Positions
    v = np.zeros([T+1, N, D])  # Velocities
    x[0], v[0] = x0, v0  # Initial conditions
    for t in range(T):
        x[t+1], v[t+1] = timestep(x[t], v[t], G, m, dt, epsilon)  # Time step
    return x, v, np.linspace(0, t_max, T+1)  # Positions, velocities, and times

def plot_trajectories(N=20, D=3, T=300, T_skip=3, epsilon=0.01):
    """Plots the trajectories of N masses in D-dimensional space."""
    # N-body simulation
    x, v, t = simulate(N, D, T=T, epsilon=epsilon)

    # Visualization
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    fig = plt.figure(figsize=(3., 3.))  # Create a new figure
    ax = plt.axes([0, 0, 1, 1])  # Add axes to figure
    ticks = np.linspace(0, 1, 6)  # Tick marks
    ax.set_xticks(ticks); ax.set_xticklabels([r'$%g$' % tick for tick in ticks])
    ax.set_yticks(ticks); ax.set_yticklabels([r'$%g$' % tick for tick in ticks])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)  # Set limits
    dj = T_skip  # Stride for time steps
    hj = dj // 2  # Half the stride
    colors = colormaps['coolwarm'](np.linspace(0, 1, T))  # Generate colors
    for i in range(N):  # For each mass
        for j in range(0,T,T_skip):  # Plot each segment of the trajectory
            ax.plot(x[j:j+dj,i,0], x[j:j+dj,i,1], c=colors[j+hj], zorder=-1)
    ax.scatter(x[-1,:,0], x[-1,:,1], c='k', s=5)  # Plot final positions
    ax.set_xlabel('x'); ax.set_ylabel('y')  # Label axes
    fig.savefig('trajectories.pdf', bbox_inches='tight', pad_inches=0., transparent=True, dpi=300)
    plt.close()

def plot_scaling(M=8, run=True):
    """Plots the scaling of the N-body simulation."""
    # Scaling test
    if run:
        import time
        Ns = 2**np.arange(1, M+1, dtype=np.int32)  # Number of masses
        print(f'Scaling test: N = {Ns}')
        runtimes = np.zeros(M)  # Initialize runtimes
        for i, N in enumerate(Ns):
            start = time.time()  # Start the timer
            simulate(N, D=3, T=300)  # N-body simulation
            stop = time.time()  # Stop the timer
            runtimes[i] = stop - start  # Store the runtime
            print(f'N = {N}, runtime = {runtimes[i]} seconds')
        print(f'Runtimes: {runtimes} seconds')
    else:
        M = 12  # Number of runtimes
        Ns = 2**np.arange(1, M+1, dtype=np.int32)  # Number of masses
        runtimes = np.array([8.27622414e-03, 1.03337765e-02, 9.79685783e-03, 1.27871037e-02,
                             3.01671028e-02, 8.14599991e-02, 2.88260937e-01, 1.30331993e+00,
                             6.27542520e+00, 2.16482849e+01, 9.10178831e+01, 3.54297569e+02]) # seconds

    # Fit a line to the log of the data
    runtimes = runtimes / runtimes[0]  # Normalize runtimes
    half = len(Ns) // 2  # Index for the last half of the points
    coeffs = np.polyfit(np.log(Ns[half:]), np.log(runtimes[half:]), 1)
    slope = coeffs[0]  # The slope of the line is the first coefficient

    # Visualization
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(4.5, 3.))
    ax = plt.axes([0, 0, 1, 1])
    ax.loglog(Ns[half:], np.exp(coeffs[1]) * Ns[half:]**slope, 'k--')  # Plot the fit line
    ax.legend([f'Slope = {slope:.2f}'])  # Add a legend with the slope
    ax.loglog(Ns, runtimes, 'o-', c='C0', ms=5, mew=0, lw=1)
    ax.set_xlabel(r'${\rm Number\ \,of\ \,Bodies}\ \ (N)$')
    ax.set_ylabel(r'${\rm Relative\ \,Runtime}\ \ (t_N / t_2)$')
    fig.savefig(f'scaling_{M}.pdf', bbox_inches='tight', pad_inches=0.025, transparent=True, dpi=300)
    plt.close()

    fig = plt.figure(figsize=(4.5, 3.))
    ax = plt.axes([0, 0, 1, 1])
    ax.semilogx(Ns[1:], [4.]*(M-1), 'k--')  # Plot the reference line
    ax.semilogx(Ns[1:], runtimes[1:]/runtimes[:-1], 'o-', c='C0', ms=5, mew=0, lw=1)
    ax.set_xlabel(r'${\rm Number\ \,of\ \,Bodies}\ \ (N)$')
    ax.set_ylabel(r'${\rm Relative\ \,Double\ \,Runtime}\ \ (t_N / t_{N/2})$')
    fig.savefig(f'doubling_{M}.pdf', bbox_inches='tight', pad_inches=0.025, transparent=True, dpi=300)
    plt.close()

if __name__ == '__main__':
    # plot_trajectories()  # Visualize an N-body simulation
    # plot_scaling(6)  # Visualize the scaling of an N-body simulation
    # plot_scaling(12)  # Visualize the scaling of an N-body simulation
    plot_scaling(12, run=False)  # Visualize the scaling of an N-body simulation
