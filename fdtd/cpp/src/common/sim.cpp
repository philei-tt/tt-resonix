#include "common/sim.hpp"
#include <cmath>

template <typename Dtype>
std::vector<Dtype> make_fd_coeffs(int m) {
    // Solve  Σ_k c_k k^(2n+1) = rhs_n,  n = 0 … m-1
    std::vector<Dtype> rhs(m, 0.0);
    rhs[0] = 0.5;

    std::vector<std::vector<Dtype>> A(m, std::vector<Dtype>(m));
    for (int n = 0; n < m; ++n) {
        for (int k = 1; k <= m; ++k) {
            A[n][k - 1] = std::pow(k, 2 * n + 1);
        }
    }

    // Gaussian elimination (tiny m, so simplicity beats libraries)
    for (int i = 0; i < m; ++i) {
        // pivot
        Dtype piv = A[i][i];
        for (int j = i; j < m; ++j) {
            A[i][j] /= piv;
        }
        rhs[i] /= piv;
        // eliminate
        for (int r = 0; r < m; ++r) {
            if (r != i) {
                Dtype f = A[r][i];
                for (int c = i; c < m; ++c) {
                    A[r][c] -= f * A[i][c];
                }
                rhs[r] -= f * rhs[i];
            }
        }
    }
    return rhs;  // length-m
}

template <typename Dtype>
Dtype gaussian_wavelet(Dtype t, Dtype f0, Dtype amp) {
    Dtype tau = 1.0 / f0, t0 = 3.0 * tau;
    return amp * std::exp(-sqr(t - t0) / sqr(tau));
}

template <typename Dtype>
Dtype ricker_wavelet(Dtype t, Dtype f0, Dtype amp) {
    Dtype a = M_PI * f0 * (t - 1.0 / f0);
    return amp * (1.0 - 2.0 * sqr(a)) * std::exp(-sqr(a));
}

template std::vector<float> make_fd_coeffs<float>(int m);
template std::vector<double> make_fd_coeffs<double>(int m);

template float gaussian_wavelet<float>(float t, float f0, float amp);
template double gaussian_wavelet<double>(double t, double f0, double amp);

template float ricker_wavelet<float>(float t, float f0, float amp);
template double ricker_wavelet<double>(double t, double f0, double amp);