#include <iostream>
#include <iomanip>    // Added for setprecision
#include <random>
#include <chrono>
#include <cmath>
#include <omp.h>

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <number_of_samples> <number_of_threads>" << std::endl;
        return 1;
    }

    int samples = std::atoi(argv[1]);    // Number of samples
    int nthreads = std::atoi(argv[2]);   // Number of threads
    omp_set_num_threads(nthreads);

    std::cout << "Number of threads: " << nthreads << std::endl;

    // Get start time
    auto t0 = omp_get_wtime();
    int count = 0;

#pragma omp parallel
    {
        // Each thread gets its own random number generator
        unsigned int seed = 1202107158 + omp_get_thread_num() * 1999;
        std::mt19937 gen(seed);  // Mersenne Twister random number generator
        std::uniform_real_distribution<double> dist(0.0, 1.0);

#pragma omp for reduction(+:count)
        for (int i = 0; i < samples; i++) {
            double x = dist(gen);
            double y = dist(gen);
            double z = x * x + y * y;
            if (z <= 1.0) {
                count++;
            }
        }
    }

    // Get end time
    auto t1 = omp_get_wtime();
    double tf = t1 - t0;

    // Estimate PI
    double PI = 4.0 * count / samples;

    // Output results
    std::cout << std::fixed << std::setprecision(5);
    std::cout << "Exact value of PI: " << M_PI << std::endl;
    std::cout << "Estimate of PI:    " << PI << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Time: " << tf << " sec." << std::endl;

    return 0;
}
