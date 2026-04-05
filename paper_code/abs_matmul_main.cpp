//PROTOTYPE: Alternative Basis Strassen — Schwartz & Vaknin
//To run:  g++ -O3 -march=native -std=c++17 -fopenmp abs_matmul_main.cpp -o abs_mm

#include "abs_matmul.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <string>
#include <vector>

using namespace abs_mm;
using HRC = std::chrono::high_resolution_clock;

void fill_random(Scalar* M, std::size_t n2, std::mt19937_64& rng) {
    std::uniform_real_distribution<Scalar> dist(-1.0, 1.0);
    for (std::size_t i = 0; i < n2; ++i) M[i] = dist(rng);
}

void print_mat(const Scalar* M, std::size_t n, const std::string& name) {
    std::cout << name << " [" << n << "x" << n << "]:\n";
    for (std::size_t i = 0; i < n; ++i) {
        std::cout << "  ";
        for (std::size_t j = 0; j < n; ++j)
            std::cout << std::setw(9) << std::fixed << std::setprecision(4)
                      << M[i*n+j] << " ";
        std::cout << "\n";
    }
}

bool correctness_test(std::size_t n, std::size_t leaf, bool verbose = false) {
    const std::size_t n2 = n * n;
    std::vector<Scalar> A(n2), B(n2), C(n2, 0), Cref(n2, 0);
    std::mt19937_64 rng(42 + n);
    fill_random(A.data(), n2, rng);
    fill_random(B.data(), n2, rng);

    Config cfg; cfg.leaf = leaf;
    matmul(A.data(), B.data(), C.data(), n, cfg);
    ref_matmul(A.data(), B.data(), Cref.data(), n);

    double err = max_abs_error(C.data(), Cref.data(), n2);
    bool ok = err < 1e-3;

    std::cout << "  n=" << std::setw(5) << n
              << "  leaf=" << std::setw(4) << leaf
              << "  max_err=" << std::scientific << std::setprecision(3) << err
              << "  " << (ok ? "PASS" : "FAIL")
              << "\n";

    if (verbose || !ok) {
        print_mat(A.data(),    n, "A");
        print_mat(B.data(),    n, "B");
        print_mat(C.data(),    n, "C_ABS");
        print_mat(Cref.data(), n, "C_ref");
    }
    return ok;
}

void benchmark(std::size_t n, std::size_t leaf, int reps = 5) {
    const std::size_t n2 = n * n;
    std::vector<Scalar> A(n2), B(n2), C(n2, 0);
    std::mt19937_64 rng(0);
    fill_random(A.data(), n2, rng);
    fill_random(B.data(), n2, rng);

    Config cfg; cfg.leaf = leaf;

    // Warm-up
    matmul(A.data(), B.data(), C.data(), n, cfg);

    double best = 1e18;
    for (int r = 0; r < reps; ++r) {
        std::fill(C.begin(), C.end(), 0.0);
        auto t0 = HRC::now();
        matmul(A.data(), B.data(), C.data(), n, cfg);
        auto t1 = HRC::now();
        double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
        best = std::min(best, ms);
    }
    // Report  GFLOP/s
    double gflops = 2.0*(double)n*(double)n*(double)n*1e-9 / (best*1e-3);
    std::cout << "  n=" << std::setw(6) << n
              << "  leaf=" << std::setw(4) << leaf
              << "  time=" << std::fixed << std::setprecision(1) << best << "ms"
              << "  eff=" << std::fixed << std::setprecision(2) << gflops << " GFLOP/s\n";
}

// ---------------------------------------------------------------------------
int main() {
    std::cout << " Alternative Basis Strassen — Schwartz & Vaknin\n";

    std::cout << "Testing things....\n";
    bool all_ok = true;

    // Tiny exact cases: n == leaf, single leaf block, no recursion
    all_ok &= correctness_test(2,  2);    // 2x2, leaf=2: direct GEMM
    all_ok &= correctness_test(4,  4);    // 4x4, leaf=4: direct GEMM
    all_ok &= correctness_test(2,  1);    // 2x2, leaf=1: 1 ABS step, leaf is 1x1

    // Small power-of-2 with recursion
    all_ok &= correctness_test(4,  2);    // 4x4,  1 ABS step, leaf=2
    all_ok &= correctness_test(8,  4);    // 8x8,  1 ABS step, leaf=4
    all_ok &= correctness_test(8,  2);    // 8x8,  2 ABS steps, leaf=2
    all_ok &= correctness_test(16, 8);    // 16x16, 1 step
    all_ok &= correctness_test(16, 4);    // 16x16, 2 steps
    all_ok &= correctness_test(16, 2);    // 16x16, 3 steps
    all_ok &= correctness_test(32, 8);    // 32x32
    all_ok &= correctness_test(64, 8);    // 64x64

    // Non-power-of-2 dimensions (padded)
    all_ok &= correctness_test(3,  2);    // padded to 4
    all_ok &= correctness_test(5,  4);    // padded to 8 (leaf=4 -> 2 steps)
    all_ok &= correctness_test(6,  4);    // padded to 8
    all_ok &= correctness_test(7,  4);    // padded to 8
    all_ok &= correctness_test(10, 8);    // padded to 16
    all_ok &= correctness_test(13, 8);    // padded to 16
    all_ok &= correctness_test(17, 8);    // padded to 32
    all_ok &= correctness_test(100, 64);  // padded to 128

    correctness_test(2, 1, true); //(,,Verbosity)

    std::cout << "\nAll tests: " << (all_ok ? "PASSED"
                                            : "FAILED") << "\n\n";

    std::cout << "Benchmarks (best of 5)";
    for (std::size_t n : {128UL, 256UL, 512UL, 1024UL}) {
        benchmark(n, 64);
    }
    // few with smaller leaf
    benchmark(512,  32);
    benchmark(1024, 32);

    return all_ok ? 0 : 1;
}
