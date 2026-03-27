#include <iostream>
#include <random>
#include <mpi.h>
#include <omp.h>

using std::cout;
using std::endl;
static const int ROOT = 0;

int main()
{
    MPI_Init(NULL, NULL); // Initialize MPI
    int rank, n_ranks; // Process rank and number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Unique process rank
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
    const int n_threads = omp_get_max_threads(); // Number of threads

    // Start timing
    double start_time = MPI_Wtime();

    int h = 0, n = 1e9;
    #pragma omp parallel 
    {
        int thread = omp_get_thread_num(); // Unique thread number
        std::mt19937 gen(thread*n_ranks + rank); // Mersenne twister engine
        std::uniform_real_distribution<> rand(0., 1.);

        #pragma omp for
        for (int i = rank; i < n; i += n_ranks)
        {
            // Get random points
            double x = rand(gen);
            double y = rand(gen);

            // Check if point is inside the circle
            if (x*x + y*y <= 1.) 
            {
                #pragma omp atomic
                ++h;
            }
        }
    }

    if (rank == ROOT)
    {
        // In the root process, use MPI_IN_PLACE to do the reduction in-place
        MPI_Reduce(MPI_IN_PLACE, &h, 1, MPI_INT, MPI_SUM, ROOT, MPI_COMM_WORLD);
        double pi = 4. * double(h) / double(n);
        cout << "Pi: " << pi << " (n = " << n << ")" << endl;
        double end_time = MPI_Wtime();
        double computation_time = end_time - start_time;
        cout << "runtime: " << computation_time << " s  (t x n: "
             << computation_time * double(n_ranks * n_threads) << " s)" << endl;
        cout << "Number of processes: " << n_ranks << endl;
        cout << "Number of threads per process: " << n_threads << endl;
    } else {
        // In non-root processes, pass the send buffer as the first argument
        MPI_Reduce(&h, NULL, 1, MPI_INT, MPI_SUM, ROOT, MPI_COMM_WORLD);
    }

    MPI_Finalize(); // Finalize MPI
    return 0;
}
