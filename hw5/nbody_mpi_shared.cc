//File for N Body problem
#include <iostream> // Standard I/O
#include <fstream> // File I/O
#include <random> // Random number generators
#include <vector> // Vector (dynamic array)
#include <tuple> // Tuple (multiple return values)
#include <chrono> // Time utilities
#include <mpi.h> // MPI (parallelism)

// Global constants
static const int D = 3; // Dimensionality
static int N = 512; // Number of particles (can be set via CLI)
static int ND = N * D; // Size of the state vectors (computed from N)
static const double G = 0.5; // Gravitational constant
static const double dt = 1e-3; // Time step size
static const int T = 300; // Number of time steps
static const double t_max = static_cast<double>(T) * dt; // Maximum time
static const double x_min = 0.; // Minimum position
static const double x_max = 1.; // Maximum position
static const double v_min = 0.; // Minimum velocity
static const double v_max = 0.; // Maximum velocity
static const double m_0 = 1.; // Mass value
static const double epsilon = 0.01; // Softening parameter
static const double epsilon2 = epsilon * epsilon; // Softening parameter^2
static int rank, n_ranks; // Process rank and number of processes
static std::vector<int> counts, displs; // Counts and displacements for MPI_Allgatherv
static std::vector<int> countsD, displsD; // State counts and displacements for MPI_Allgatherv
static int N_beg, N_end, N_local; // Mass range for each process [N_beg, N_end)
static int ND_beg, ND_end, ND_local; // State vector range for each process [ND_beg, ND_end)
// Shared memory for masses, positions, velocities, and accelerations
static double *m, *x, *v, *a, *x_next, *v_next; // Shared memory
static MPI_Win win_m, win_x, win_v, win_a, win_x_next, win_v_next; // Shared windows

// Note that epsilon must be greater than zero!
using Vec = std::vector<double>; // Vector type
using Vecs = std::vector<Vec>; // Vector of vectors type

// Random number generator
static std::mt19937 gen; // Mersenne twister engine
static std::uniform_real_distribution<> ran(0., 1.); // Uniform distribution

// Set up parallelism
void setup_parallelism() {
    MPI_Init(NULL, NULL); // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Unique process rank
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
    
    // Print rank information
    std::cout << "Rank " << rank << " of " << n_ranks << " ranks" << std::endl;

    // Get the current time and convert it to an integer
    auto now = std::chrono::high_resolution_clock::now();
    auto now_cast = std::chrono::time_point_cast<std::chrono::microseconds>(now);
    auto now_int = now_cast.time_since_epoch().count();

    // Pure MPI version
    gen.seed(now_int ^ rank); // Seed the random number generator

    // Divide the masses among the processes (needed for MPI_Allgatherv)
    counts.resize(n_ranks); // Counts for each process
    displs.resize(n_ranks); // Displacements for each process
    countsD.resize(n_ranks); // State counts for each process
    displsD.resize(n_ranks); // State displacements for each process
    const int remainder = N % n_ranks; // Remainder of the division
    for (int i = 0; i < n_ranks; ++i) {
        counts[i] = N / n_ranks; // Divide the masses among the processes
        displs[i] = i * counts[i]; // Displacements where each segment begins
        if (i < remainder) {
        counts[i] += 1; // Correct the count
        displs[i] += i; // Correct the displacement
        }   else {
            displs[i] += remainder; // Correct the displacement
        }
        countsD[i] = counts[i] * D; // State counts for each process
        displsD[i] = displs[i] * D; // State displacements for each process
    }

    // Set up the local mass ranges
    N_beg = displs[rank]; // Mass range for each process [N_beg, N_end)
    N_end = N_beg + counts[rank]; // Mass range for each process [N_beg, N_end)
    ND_beg = N_beg * D; // State vector range for each process [ND_beg, ND_end)
    ND_end = N_end * D; // State vector range for each process [ND_beg, ND_end)
    N_local = N_end - N_beg; // Local number of masses
    ND_local = ND_end - ND_beg; // Local size of the state vectors
    // Allocate shared memory for positions, velocities, and accelerations
     if (rank == 0) {
        MPI_Win_allocate_shared(N * sizeof(double), sizeof(double),
                                MPI_INFO_NULL, MPI_COMM_WORLD, &m, &win_m);
        MPI_Win_allocate_shared(ND * sizeof(double), sizeof(double),
                                MPI_INFO_NULL, MPI_COMM_WORLD, &x, &win_x);
        MPI_Win_allocate_shared(ND * sizeof(double), sizeof(double),
                                MPI_INFO_NULL, MPI_COMM_WORLD, &v, &win_v);
        MPI_Win_allocate_shared(ND * sizeof(double), sizeof(double),
                                MPI_INFO_NULL, MPI_COMM_WORLD, &a, &win_a);
        MPI_Win_allocate_shared(ND * sizeof(double), sizeof(double),
                                MPI_INFO_NULL, MPI_COMM_WORLD, &x_next, &win_x_next);
        MPI_Win_allocate_shared(ND * sizeof(double), sizeof(double),
                                MPI_INFO_NULL, MPI_COMM_WORLD, &v_next, &win_v_next);
    } else {
        int disp_unit;
        MPI_Aint size;

        MPI_Win_allocate_shared(0, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &m, &win_m);
        MPI_Win_shared_query(win_m, 0, &size, &disp_unit, &m);

        MPI_Win_allocate_shared(0, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &x, &win_x);
        MPI_Win_shared_query(win_x, 0, &size, &disp_unit, &x);

        MPI_Win_allocate_shared(0, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &v, &win_v);
        MPI_Win_shared_query(win_v, 0, &size, &disp_unit, &v);

        MPI_Win_allocate_shared(0, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &a, &win_a);
        MPI_Win_shared_query(win_a, 0, &size, &disp_unit, &a);

        MPI_Win_allocate_shared(0, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &x_next, &win_x_next);
        MPI_Win_shared_query(win_x_next, 0, &size, &disp_unit, &x_next);

        MPI_Win_allocate_shared(0, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &v_next, &win_v_next);
        MPI_Win_shared_query(win_v_next, 0, &size, &disp_unit, &v_next);
    }
    // initialize the masses
     for (int i = N_beg; i < N_end; ++i)
        m[i] = m_0;
    MPI_Barrier(MPI_COMM_WORLD);
}

// Print a vector to a file
template <typename T>
void save(const std::vector<T>& vec, const std::string& filename,
          const std::string& header = "") {
    std::ofstream file(filename); // Open the file
    if (file.is_open()) { // Check for successful opening
        if (!header.empty())
        {
            file << "# " << header << std::endl; // Write the header
        }
        for (const auto& elem : vec)
        {
            file << elem << " "; // Write each element
        }
        file << std::endl; // Write a newline
        file.close(); // Close the file
    }   
    else {
        std::cerr << "Unable to open file " << filename << std::endl;
    }
}

// Generate random initial conditions for N masses
void initial_conditions() {
    const double dx = x_max - x_min;
    const double dv = v_max - v_min;

    for (int i = ND_beg; i < ND_end; ++i) {
        x[i] = ran(gen) * dx + x_min;
        v[i] = ran(gen) * dv + v_min;
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

// Compute the acceleration of all masses
// a_i = G * sum_{ji} m_j * (x_j - x_i) / |x_j - x_i|^3
void acceleration() {
    for (int i = ND_beg; i < ND_end; ++i)
        a[i] = 0.0;

    for (int i = N_beg; i < N_end; ++i) {
        const int iD = i * D;
        double dx[D];

        for (int j = 0; j < N; ++j) {
            const int jD = j * D;
            double dx2 = epsilon2;

            for (int k = 0; k < D; ++k) {
                dx[k] = x[jD + k] - x[iD + k];
                dx2 += dx[k] * dx[k];
            }

            const double Gm_dx3 = G * m[j] / (dx2 * sqrt(dx2));
            for (int k = 0; k < D; ++k)
                a[iD + k] += Gm_dx3 * dx[k];
        }
    }
}

// Compute the next position and velocity for all masses
void timestep() {
    acceleration();

    for (int i = ND_beg; i < ND_end; ++i) {
        v_next[i] = a[i] * dt + v[i];
        x_next[i] = v_next[i] * dt + x[i];
    }

    std::swap(x, x_next);
    std::swap(v, v_next);

    MPI_Barrier(MPI_COMM_WORLD);
}

// Main function
int main(int argc, char** argv) {
    // Parse command-line arguments for N
    if (argc > 1) {
        N = std::atoi(argv[1]);
        ND = N * D;
    }
    
    // Initialize MPI before any other MPI calls
    setup_parallelism();
    MPI_Barrier(MPI_COMM_WORLD);
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Keep only lightweight bookkeeping locally; particle state lives in shared memory.
    Vec t(T+1); // Time points
    for (int i = 0; i <= T; ++i)
        t[i] = double(i) * dt; // Time points
    initial_conditions(); // Set up initial conditions

    // Simulate the motion of N masses in D-dimensional space
    for (int n = 0; n < T; ++n)
        timestep(); // Time step

    // Sum kinetic energy over the current shared velocity array.
    double KE_local = 0.;
    for (int i = N_beg; i < N_end; ++i) {
        double v2 = 0.; // Velocity magnitude
        for (int j = 0; j < D; ++j) {
            const int k = i * D + j; // Flatten the index
            v2 += v[k] * v[k]; // Velocity magnitude
        }
        KE_local += 0.5 * m[i] * v2; // Kinetic energy
    }

    double KE_total = KE_local;
    if (rank == 0) {
        MPI_Reduce(MPI_IN_PLACE, &KE_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    } else {
        MPI_Reduce(&KE_local, NULL, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    // Stop timing
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.;

    if (rank == 0) {
        std::cout << "Total Kinetic Energy = " << KE_total << std::endl;
        std::cout << "Runtime = " << elapsed << " s for N = " << N << std::endl;
    }
    
    MPI_Win_free(&win_v_next);
    MPI_Win_free(&win_x_next);
    MPI_Win_free(&win_a);
    MPI_Win_free(&win_v);
    MPI_Win_free(&win_x);
    MPI_Win_free(&win_m);
    MPI_Finalize(); // Finalize MPI
    return 0;
}