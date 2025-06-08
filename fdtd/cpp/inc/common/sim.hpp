#pragma once

#include <vector>

template <typename T>
inline T sqr(T x) { return x * x; }

std::vector<double> make_fd_coeffs(int m);

double gaussian_wavelet(double t, double f0, double amp);
double ricker_wavelet(double t, double f0, double amp);
