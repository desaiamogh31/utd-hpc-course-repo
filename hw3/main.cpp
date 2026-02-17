#include <iostream>
#include <random>
#include <cstdlib> // for atoi
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " N\n";
        return 1;
        }
    const int n = std::atoi(argv[1]);
    std::random_device rd; // Random seed from hardware
    std::mt19937 gen(rd()); // Mersenne twister engine
    std::uniform_real_distribution<> rand(0., 1.);
    int h = 0;
    for (int i = 0; i < n; ++i) {
        // Get random points
        const double x = rand(gen);
        const double y = rand(gen);// Check if point is inside the circle
        if (x*x + y*y <= 1.) h++;
        }
    double pi = 4. * double(h) / double(n);
        std::cout << "n=" << n << ", pi=" << pi << std::endl;
    return 0;
}