// Alternative Basis Strassen (Schwartz & Vaknin, SIAM 2023)

// Prototype: eq 3.1 + Appendix B of the paper.
// Basis transforms phi_opt / nu_opt^{-1}: 2 ops each, in-place.
// Bilinear phase: 7 multiplications, 12 additions (Appendix B, naive schedule).
// Recursive structure: l steps of ABS, then naive GEMM on b×b leaf blocks.
// Data layout: flat row-major throughout (simples).

#pragma once

#include <cstddef>
#include <cstring>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <memory>
#include <vector>
#include <stdexcept>

namespace abs_mm {
using Scalar = double;

// Default leaf-block size, leaf is sent to the GEMM kernel, naive case
static constexpr std::size_t DEFAULT_LEAF = 64;


// Naive leaf Gemm, scope for improvements here.
inline void leaf_gemm(const Scalar* __restrict__ A,
                      const Scalar* __restrict__ B,
                      Scalar* __restrict__ C,
                      std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < n; ++k) {
            const Scalar a = A[i * n + k];
            if (a == 0.0) continue;
#pragma omp simd
            for (std::size_t j = 0; j < n; ++j)
                C[i * n + j] += a * B[k * n + j];
        }
    }
}

// For generating sub-views of the larger matrix

// Pointer to quadrant (qr, qc) of an n×n row-major matrix with stride n.
inline Scalar* Q(Scalar* M, std::size_t n, std::size_t half,
                 int qr, int qc) noexcept {
    return M + qr * half * n + qc * half;
}
inline const Scalar* Q(const Scalar* M, std::size_t n, std::size_t half,
                       int qr, int qc) noexcept {
    return M + qr * half * n + qc * half;
}

// dst = a + b
inline void blk_add(const Scalar* __restrict__ a,
                    const Scalar* __restrict__ b,
                    Scalar* __restrict__ dst,
                    std::size_t h, std::size_t stride) noexcept {
    for (std::size_t i = 0; i < h; ++i)
#pragma omp simd
        for (std::size_t j = 0; j < h; ++j)
            dst[i*stride+j] = a[i*stride+j] + b[i*stride+j];
}

// dst = a - b
inline void blk_sub(const Scalar* __restrict__ a,
                    const Scalar* __restrict__ b,
                    Scalar* __restrict__ dst,
                    std::size_t h, std::size_t stride) noexcept {
    for (std::size_t i = 0; i < h; ++i)
#pragma omp simd
        for (std::size_t j = 0; j < h; ++j)
            dst[i*stride+j] = a[i*stride+j] - b[i*stride+j];
}

// dst += a
inline void blk_add_ip(Scalar* __restrict__ dst,
                       const Scalar* __restrict__ a,
                       std::size_t h, std::size_t stride) noexcept {
    for (std::size_t i = 0; i < h; ++i)
#pragma omp simd
        for (std::size_t j = 0; j < h; ++j)
            dst[i*stride+j] += a[i*stride+j];
}

// dst -= a
inline void blk_sub_ip(Scalar* __restrict__ dst,
                       const Scalar* __restrict__ a,
                       std::size_t h, std::size_t stride) noexcept {
    for (std::size_t i = 0; i < h; ++i)
#pragma omp simd
        for (std::size_t j = 0; j < h; ++j)
            dst[i*stride+j] -= a[i*stride+j];
}

// dst = b - a
inline void blk_neg_add(const Scalar* __restrict__ a,
                        const Scalar* __restrict__ b,
                        Scalar* __restrict__ dst,
                        std::size_t h, std::size_t stride) noexcept {
    for (std::size_t i = 0; i < h; ++i)
#pragma omp simd
        for (std::size_t j = 0; j < h; ++j)
            dst[i*stride+j] = -a[i*stride+j] + b[i*stride+j];
}

// Copy a sub-block into a contiguous flat buffer (stride h)
inline void blk_copy_out(const Scalar* __restrict__ src,
                         Scalar* __restrict__ dst,
                         std::size_t h, std::size_t stride) noexcept {
    for (std::size_t i = 0; i < h; ++i)
        std::memcpy(dst + i*h, src + i*stride, h * sizeof(Scalar));
}

// Copy a contiguous flat buffer into a sub-block (stride stride)
inline void blk_copy_in(const Scalar* __restrict__ src,
                        Scalar* __restrict__ dst,
                        std::size_t h, std::size_t stride) noexcept {
    for (std::size_t i = 0; i < h; ++i)
        std::memcpy(dst + i*stride, src + i*h, h * sizeof(Scalar));
}

// Zero a sub-block
inline void blk_zero(Scalar* dst, std::size_t h, std::size_t stride) noexcept {
    for (std::size_t i = 0; i < h; ++i)
        std::memset(dst + i*stride, 0, h * sizeof(Scalar));
}

// ------------------------------------------------------------------------
// Basis transforms (in-place on sub-matrices with stride)
//  Paper 2, Equation (3.1) and Appendix B:
//
//   phi_opt applied to a 2×2 block matrix [[A11,A12],[A21,A22]]:
//       A11 unchanged
//       A12 unchanged
//       A21 unchanged
//       A22 <-- A12 - A21 + A22          (2 ops: one sub, one add)
//
//   nu_opt^{-1} applied to [[C11,C12],[C21,C22]]:
//       C11 unchanged
//       C12 <-- C12 - C22                (1 op)
//       C21 <-- -C21 + C22               (1 op, = C22 - C21)
//       C22 unchanged
//
// --------------------------------------------------------------------------



// NOTE: This is a recursive interpretation of the basis transform.
// Need to check against the paper's exact recursion definition.
// Stride-aware phi transform: M is an (n×n) sub-block with row stride `stride`.
void apply_phi_strided(Scalar* M, std::size_t n, std::size_t stride,
                       std::size_t leaf_n) noexcept {
    if (n <= 1) return;   // base: single element, phi is identity

    const std::size_t h = n / 2;

    // Quadrant pointers (stride = stride throughout)
    Scalar* A11 = M + 0*h*stride + 0*h;
    Scalar* A12 = M + 0*h*stride + 1*h;
    Scalar* A21 = M + 1*h*stride + 0*h;
    Scalar* A22 = M + 1*h*stride + 1*h;

    // A22 <- A12 - A21 + A22
    for (std::size_t i = 0; i < h; ++i)
#pragma omp simd
        for (std::size_t j = 0; j < h; ++j)
            A22[i*stride+j] = A12[i*stride+j] - A21[i*stride+j] + A22[i*stride+j];

    // Recurse into each quadrant only if we haven't reached the leaf level
    if (h >= leaf_n && h > 1) {
        apply_phi_strided(A11, h, stride, leaf_n);
        apply_phi_strided(A12, h, stride, leaf_n);
        apply_phi_strided(A21, h, stride, leaf_n);
        apply_phi_strided(A22, h, stride, leaf_n);
    }
}

// recursion stops when n == leaf_n.
void apply_phi(Scalar* M, std::size_t n, std::size_t leaf_n) noexcept {
    apply_phi_strided(M, n, n, leaf_n);


//     if (n <= 1) return;          // scalar: identity (no block structure)

//     const std::size_t h = n / 2;

//     // Pointers to the four quadrants (stride = n throughout)
//     Scalar* A11 = Q(M, n, h, 0, 0);
//     Scalar* A12 = Q(M, n, h, 0, 1);
//     Scalar* A21 = Q(M, n, h, 1, 0);
//     Scalar* A22 = Q(M, n, h, 1, 1);

//     // A22 <- A12 - A21 + A22  (merged into one pass)
//     for (std::size_t i = 0; i < h; ++i)
// #pragma omp simd
//         for (std::size_t j = 0; j < h; ++j)
//             A22[i*n+j] = A12[i*n+j] - A21[i*n+j] + A22[i*n+j];

//     if (h > 1 && h >= leaf_n) {
//     }
//     // We do NOT recurse further here; see apply_phi_strided.
}

// Recursive prototype of the output-basis inverse transform.
void apply_nu_inv_strided(Scalar* M, std::size_t n, std::size_t stride,
                          std::size_t leaf_n) noexcept {
    if (n <= 1) return;

    const std::size_t h = n / 2;

    Scalar* C11 = M + 0*h*stride + 0*h;
    Scalar* C12 = M + 0*h*stride + 1*h;
    Scalar* C21 = M + 1*h*stride + 0*h;
    Scalar* C22 = M + 1*h*stride + 1*h;

    // C12 <- C12 - C22
    for (std::size_t i = 0; i < h; ++i)
#pragma omp simd
        for (std::size_t j = 0; j < h; ++j)
            C12[i*stride+j] -= C22[i*stride+j];

    // C21 <- -C21 + C22
    for (std::size_t i = 0; i < h; ++i)
#pragma omp simd
        for (std::size_t j = 0; j < h; ++j)
            C21[i*stride+j] = -C21[i*stride+j] + C22[i*stride+j];

    // Recurse
    if (h >= leaf_n && h > 1) {
        apply_nu_inv_strided(C11, h, stride, leaf_n);
        apply_nu_inv_strided(C12, h, stride, leaf_n);
        apply_nu_inv_strided(C21, h, stride, leaf_n);
        apply_nu_inv_strided(C22, h, stride, leaf_n);
    }
}



// Computes C = A * B where A, B, C are n×n, row-major, stride n.
//
//   m1 = A11 * B11
//   m2 = A12 * B21
//   m3 = A21 * (B22 - B11)     temp_R = B22 - B11
//   m4 = A22 * B22
//   m5 = (A21 + A22) * (B21 + B22)   temp_L = A21+A22, temp_R = B21+B22
//   m6 = (A22 - A12) * (B22 - B12)   temp_L = A22-A12, temp_R = B22-B12
//   m7 = (-A11 + A22) * B12     temp_L = -A11+A22
//
// Output assembly (Appendix B):
//   C11 = m1 + m2
//   C12 = m5 - m7
//   C21 = m3 + m6
//   C22 = m5 + m6 - m2 - m4
//
// NOTE: Prototype implementation: uses explicit temporaries for clarity.
// Production code should use a workspace arena and temporary reuse.
void bilinear_phase(const Scalar* __restrict__ A,
                    const Scalar* __restrict__ B,
                    Scalar* __restrict__ C,
                    std::size_t n,
                    std::size_t leaf_n) {
    if (n <= leaf_n) {
        leaf_gemm(A, B, C, n);
        return;
    }

    const std::size_t h  = n / 2;
    const std::size_t h2 = h * h;

    // Input quadrant pointers (stride = n)
    const Scalar* A11 = Q(A, n, h, 0, 0);
    const Scalar* A12 = Q(A, n, h, 0, 1);
    const Scalar* A21 = Q(A, n, h, 1, 0);
    const Scalar* A22 = Q(A, n, h, 1, 1);

    const Scalar* B11 = Q(B, n, h, 0, 0);
    const Scalar* B12 = Q(B, n, h, 0, 1);
    const Scalar* B21 = Q(B, n, h, 1, 0);
    const Scalar* B22 = Q(B, n, h, 1, 1);

    // Output quadrant pointers (stride = n)
    Scalar* C11 = Q(C, n, h, 0, 0);
    Scalar* C12 = Q(C, n, h, 0, 1);
    Scalar* C21 = Q(C, n, h, 1, 0);
    Scalar* C22 = Q(C, n, h, 1, 1);

    // Using std::vector for automatic cleanup; in production, use a workspace arena.
    std::vector<Scalar> _m1(h2,0), _m2(h2,0), _m3(h2,0), _m4(h2,0),
                         _m5(h2,0), _m6(h2,0), _m7(h2,0),
                         _tL(h2),   _tR(h2);
    Scalar* m1 = _m1.data(); Scalar* m2 = _m2.data();
    Scalar* m3 = _m3.data(); Scalar* m4 = _m4.data();
    Scalar* m5 = _m5.data(); Scalar* m6 = _m6.data();
    Scalar* m7 = _m7.data();
    Scalar* tL = _tL.data(); Scalar* tR = _tR.data();

    // Helper: copy a strided h×h sub-block into a flat h×h buffer
    auto extract = [&](const Scalar* src, Scalar* dst) {
        for (std::size_t i = 0; i < h; ++i)
            std::memcpy(dst + i*h, src + i*n, h*sizeof(Scalar));
    };

    // Helper: add a strided source into a flat buffer: dst += src (strided)
    auto extract_add = [&](const Scalar* src_a, const Scalar* src_b, Scalar* dst) {
        // dst[i*h+j] = src_a[i*n+j] + src_b[i*n+j]
        for (std::size_t i = 0; i < h; ++i)
#pragma omp simd
            for (std::size_t j = 0; j < h; ++j)
                dst[i*h+j] = src_a[i*n+j] + src_b[i*n+j];
    };

    auto extract_sub = [&](const Scalar* src_a, const Scalar* src_b, Scalar* dst) {
        for (std::size_t i = 0; i < h; ++i)
#pragma omp simd
            for (std::size_t j = 0; j < h; ++j)
                dst[i*h+j] = src_a[i*n+j] - src_b[i*n+j];
    };

    auto extract_neg_add = [&](const Scalar* src_a, const Scalar* src_b, Scalar* dst) {
        // dst = -src_a + src_b
        for (std::size_t i = 0; i < h; ++i)
#pragma omp simd
            for (std::size_t j = 0; j < h; ++j)
                dst[i*h+j] = -src_a[i*n+j] + src_b[i*n+j];
    };

    // m1 = A11 * B11
    extract(A11, tL);   extract(B11, tR);
    bilinear_phase(tL, tR, m1, h, leaf_n);

    // m2 = A12 * B21
    extract(A12, tL);   extract(B21, tR);
    bilinear_phase(tL, tR, m2, h, leaf_n);

    // m3 = A21 * (B22 - B11)
    extract(A21, tL);
    extract_sub(B22, B11, tR);
    bilinear_phase(tL, tR, m3, h, leaf_n);

    // m4 = A22 * B22
    extract(A22, tL);   extract(B22, tR);
    bilinear_phase(tL, tR, m4, h, leaf_n);

    // m5 = (A21 + A22) * (B21 + B22)
    extract_add(A21, A22, tL);
    extract_add(B21, B22, tR);
    bilinear_phase(tL, tR, m5, h, leaf_n);

    // m6 = (A22 - A12) * (B22 - B12)
    extract_sub(A22, A12, tL);
    extract_sub(B22, B12, tR);
    bilinear_phase(tL, tR, m6, h, leaf_n);

    // m7 = (-A11 + A22) * B12
    extract_neg_add(A11, A22, tL);
    extract(B12, tR);
    bilinear_phase(tL, tR, m7, h, leaf_n);

    // Output assembly
    // C11 = m1 + m2
    for (std::size_t i = 0; i < h; ++i)
#pragma omp simd
        for (std::size_t j = 0; j < h; ++j)
            C11[i*n+j] = m1[i*h+j] + m2[i*h+j];

    // C12 = m5 - m7
    for (std::size_t i = 0; i < h; ++i)
#pragma omp simd
        for (std::size_t j = 0; j < h; ++j)
            C12[i*n+j] = m5[i*h+j] - m7[i*h+j];

    // C21 = m3 + m6
    for (std::size_t i = 0; i < h; ++i)
#pragma omp simd
        for (std::size_t j = 0; j < h; ++j)
            C21[i*n+j] = m3[i*h+j] + m6[i*h+j];

    // C22 = m5 + m6 - m2 - m4
    for (std::size_t i = 0; i < h; ++i)
#pragma omp simd
        for (std::size_t j = 0; j < h; ++j)
            C22[i*n+j] = m5[i*h+j] + m6[i*h+j] - m2[i*h+j] - m4[i*h+j];
}

inline std::size_t padded_dim(std::size_t n, std::size_t leaf_n) noexcept {
    std::size_t p = leaf_n;
    while (p < n) p *= 2;
    return p;
}

// Copy an n×n row-major matrix into the top-left corner of an np×np
inline void pad_matrix(const Scalar* src, std::size_t n,
                       Scalar* dst, std::size_t np) noexcept {
    // Zero-fill destination
    std::memset(dst, 0, np * np * sizeof(Scalar));
    // Copy rows
    for (std::size_t i = 0; i < n; ++i)
        std::memcpy(dst + i * np, src + i * n, n * sizeof(Scalar));
}

// Extract the top-left n×n sub-block of an np×np matrix into dst (n×n).
inline void unpad_matrix(const Scalar* src, std::size_t np,
                         Scalar* dst, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i)
        std::memcpy(dst + i * n, src + i * np, n * sizeof(Scalar));
}


struct Config {
    std::size_t leaf = DEFAULT_LEAF;   // Leaf block size (must be power of 2)
    // depth = 0 means: recurse as many times as needed until leaf is reached.
};

// Multiply C = A * B, all n×n, row-major.
// C is OVERWRITTEN (not accumulated).
// A and B are not modified.
void matmul(const Scalar* A, const Scalar* B, Scalar* C,
            std::size_t n, Config cfg = Config{}) {

    if (n == 0) return;

    const std::size_t leaf = cfg.leaf;

    if (leaf == 0 || (leaf & (leaf - 1)) != 0)
        throw std::invalid_argument("leaf must be a non-zero power of 2");

    // Pad to next power-of-2 multiple of leaf
    const std::size_t np = padded_dim(n, leaf);

    // Allocate padded matrices
    std::vector<Scalar> Ap(np * np, 0.0);
    std::vector<Scalar> Bp(np * np, 0.0);
    std::vector<Scalar> Cp(np * np, 0.0);

    pad_matrix(A, n, Ap.data(), np);
    pad_matrix(B, n, Bp.data(), np);

    // Apply phi_opt to A and B (in-place)
    // Only recurse if np > leaf (otherwise it's a single leaf block)
    if (np > leaf) {
        apply_phi_strided(Ap.data(), np, np, leaf);
        apply_phi_strided(Bp.data(), np, np, leaf);
    }

    // Bilinear phase: Cp = Ap * Bp (in the nu basis)
    bilinear_phase(Ap.data(), Bp.data(), Cp.data(), np, leaf);

    // Apply nu_opt^{-1} to Cp (in-place)
    if (np > leaf) {
        apply_nu_inv_strided(Cp.data(), np, np, leaf);
    }

    // Extract result
    unpad_matrix(Cp.data(), np, C, n);
}

// Naive O(n^3) multiply for verification
void ref_matmul(const Scalar* A, const Scalar* B, Scalar* C, std::size_t n) {
    std::memset(C, 0, n * n * sizeof(Scalar));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t k = 0; k < n; ++k) {
            const Scalar a = A[i*n+k];
#pragma omp simd
            for (std::size_t j = 0; j < n; ++j)
                C[i*n+j] += a * B[k*n+j];
        }
}

double max_abs_error(const Scalar* C, const Scalar* Cref, std::size_t n2) {
    double err = 0.0;
    for (std::size_t i = 0; i < n2; ++i)
        err = std::max(err, std::abs((double)(C[i] - Cref[i])));
    return err;
}

}
