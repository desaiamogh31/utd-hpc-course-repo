//File for N Body problem
#include <iostream> // Standard I/O
#include <fstream> // File I/O
#include <random> // Random number generators
#include <vector> // Vector (dynamic array)
#include <tuple> // Tuple (multiple return values)
#include <chrono> // Time utilities
#include <mpi.h> // MPI (parallelism)
#include <omp.h> // OpenMP (parallelism)

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
static int thread; // Unique thread number

// Note that epsilon must be greater than zero!
using Vec = std::vector<double>; // Vector type
using Vecs = std::vector<Vec>; // Vector of vectors type

// Random number generator
static std::mt19937 gen; // Mersenne twister engine
static std::uniform_real_distribution<> ran(0., 1.); // Uniform distribution
#pragma omp threadprivate(thread, gen, ran)
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
std::tuple<Vec, Vec> initial_conditions() {
    Vec x(ND), v(ND); // Allocate memory
    const double dx = x_max - x_min; // Position range
    const double dv = v_max - v_min; // Velocity range
    #pragma omp parallel for
    for (int i = ND_beg; i < ND_end; ++i) {
        x[i] = ran(gen) * dx + x_min; // Random initial positions
        v[i] = ran(gen) * dv + v_min; // Random initial velocities
    }
    if (n_ranks > 1) { // More than one process
        // Gather the initial positions and velocities
        MPI_Allgatherv(x.data() + ND_beg, ND_local, MPI_DOUBLE, x.data(),
                countsD.data(), displsD.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(v.data() + ND_beg, ND_local, MPI_DOUBLE, v.data(),
                countsD.data(), displsD.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    }
    return {x, v}; // Positions and velocities
}

// Compute the acceleration of all masses
// a_i = G * sum_{ji} m_j * (x_j - x_i) / |x_j - x_i|^3
Vec acceleration(const Vec& x, const Vec& m) {
    Vec a(ND); // Accelerations
    #pragma omp parallel for
    for (int i = N_beg; i < N_end; ++i) {
        const int iD = i * D; // Flatten the index
        double dx[D]; // Difference in position
        for (int j = 0; j < N; ++j) {
            const int jD = j * D; // Flatten the index
            double dx2 = epsilon2; // Distance^2 (softened)
            for (int k = 0; k < D; ++k) {
                dx[k] = x[jD + k] - x[iD + k]; // Difference in position
                dx2 += dx[k] * dx[k]; // Distance^2
                }
            const double Gm_dx3 = G * m[j] / (dx2 * sqrt(dx2)); // G * m_j / |dx|^3
            for (int k = 0; k < D; ++k) {
                const int iDk = iD + k; // Flatten the index
                a[iDk] += Gm_dx3 * dx[k]; // Acceleration
            }
        }
    }
    return a; // Accelerations
}

// Compute the next position and velocity for all masses
std::tuple<Vec, Vec> timestep(const Vec& x0, const Vec& v0, const Vec& m) {
    Vec a0 = acceleration(x0, m); // Calculate particle accelerations
    Vec x1(ND), v1(ND); // Allocate memory
    #pragma omp parallel for
    for (int i = ND_beg; i < ND_end; ++i) {
        v1[i] = a0[i] * dt + v0[i]; // New velocity
        x1[i] = v1[i] * dt + x0[i]; // New position
        }
    if(n_ranks > 1) { // More than one process
        // Gather the new positions and velocities
        MPI_Allgatherv(x1.data() + ND_beg, ND_local, MPI_DOUBLE, x1.data(),
                countsD.data(), displsD.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(v1.data() + ND_beg, ND_local, MPI_DOUBLE, v1.data(),
                countsD.data(), displsD.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    }
    return {x1, v1}; // New positions and velocities
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
    
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Prepare vectors for time points, masses, positions, velocities, and kinetic energy
    Vec t(T+1); // Time points
    #pragma omp parallel for
    for (int i = 0; i <= T; ++i)
        t[i] = double(i) * dt; // Time points
    Vec m(N, m_0); // Masses (all equal)
    Vecs x(T+1), v(T+1); // Positions and velocities
// Hybrid MPI+OpenMP version
    #pragma omp parallel
    {
        thread = omp_get_thread_num(); // Unique thread number
        gen.seed( (thread * n_ranks + rank)); // Seed the random number generator
    }
    std::tie(x[0], v[0]) = initial_conditions(); // Set up initial conditions

    // Simulate the motion of N masses in D-dimensional space
    for (int n = 0; n < T; ++n)
        std::tie(x[n+1], v[n+1]) = timestep(x[n], v[n], m); // Time step

    // Calculate the total kinetic energy of the system
    Vec KE(T+1); // Kinetic energy
    
    for (int n = 0; n <= T; ++n) {
        double KE_n = 0.; // Kinetic energy
        auto &v_n = v[n]; // Velocities
        #pragma omp parallel for reduction(+:KE_n)
        for (int i = N_beg; i < N_end; ++i) {
            double v2 = 0.; // Velocity magnitude
            for (int j = 0; j < D; ++j) {
                const int k = i * D + j; // Flatten the index
                v2 += v_n[k] * v_n[k]; // Velocity magnitude
            }
            KE_n += 0.5 * m[i] * v2; // Kinetic energy
        }
        KE[n] = KE_n; // Kinetic energy
    }
    

    // Print the vector to the specified file
    //save(KE, "KE_" + std::to_string(N) + ".txt", "Kinetic Energy");
    //save(t, "time_" + std::to_string(N) + ".txt", "Time");

    // Output the results
    std::cout << "Total Kinetic Energy = [" << KE[0];
    const int T_skip = T / 50; // Skip every T_skip time steps
    for (int n = 1; n <= T; n += T_skip)
        std::cout << ", " << KE[n];
    std::cout << "]" << std::endl;

    if (rank == 0) {
        // Reduce the kinetic energies
        MPI_Reduce(MPI_IN_PLACE, KE.data(), T+1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        // Protected save and print logic ...
    } 
    else {
        // Send the kinetic energies
        MPI_Reduce(KE.data(), NULL, T+1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        }
    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.;
    std::cout << "Runtime = " << elapsed << " s for N = " << N << std::endl;
    
    MPI_Finalize(); // Finalize MPI
    return 0;
}