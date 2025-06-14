#pragma once

#include <vector>

template <typename T>
inline T sqr(T x) {
    return x * x;
}

template <typename Dtype>
std::vector<Dtype> make_fd_coeffs(int m);

template <typename Dtype>
Dtype gaussian_wavelet(Dtype t, Dtype f0, Dtype amp);

template <typename Dtype>
Dtype ricker_wavelet(Dtype t, Dtype f0, Dtype amp);
