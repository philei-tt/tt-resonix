// fdtd2d.cpp  –  single-threaded 2-D acoustic FDTD, C++17
//
// Compile via the accompanying CMakeLists.txt.
// Usage:  ./fdtd2d <config.json>
//
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>
#include <chrono>
#include <fmt/core.h>

#include <cnpy.h>             // third-party/cnpy
#include <nlohmann/json.hpp>  // single-header nlohmann/json

#include "common/config.hpp"
#include "common/sim.hpp"

using std::size_t;

using Dtype = float;  // data type for the simulation

// -----------------------------------------------------------------------------
// Simulation
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " config.json\n";
        return 1;
    }
    auto C = Config<Dtype>::read_config(argv[1]);
    fmt::print("{}\n", C);

    const int halo = C.m;
    const size_t Ny = C.ny + 2 * halo;
    const size_t Nx = C.nx + 2 * halo;
    const size_t stride = Nx;
    const size_t default_stride = stride;

    const Dtype dt = 0.4 * std::min(C.dx, C.dy) / (C.c * std::sqrt(2.0));

    std::vector<Dtype> p(Ny * Nx, 0.0f);
    std::vector<Dtype> vx(Ny * Nx, 0.0f);
    std::vector<Dtype> vy(Ny * Nx, 0.0f);
    std::vector<Dtype> div(C.ny * C.nx, 0.0f);


    auto idx_s = [&](size_t i, size_t j, const size_t stride) {
        return i * stride + j;  // C-order indexing
    };

    auto idx = [&](size_t i, size_t j) { return i * default_stride + j; };

    auto src_pos = idx(C.sy, C.sx);

    const int n_frames = C.n_steps / C.output_every + 1;
    std::vector<Dtype> frames;
    frames.reserve(static_cast<size_t>(n_frames) * C.ny * C.nx);

    auto coeff = C.coeffs;  // copy for brevity

    // Pre-compute reciprocal distances
    const Dtype inv_dx = 1.0f / C.dx;
    const Dtype inv_dy = 1.0f / C.dy;
    const Dtype rhoc2 = static_cast<Dtype>(C.rho * C.c * C.c);
    const Dtype inv_rho = static_cast<Dtype>(1.0 / C.rho);

    // ---------------------------------------------------------------------
    // output frame 0  (pressure is all zeros)
    for (int i = 0; i < C.ny; ++i) {
        for (int j = 0; j < C.nx; ++j) {
            frames.push_back(p[idx(i + halo, j + halo)]);
        }
    }

    auto update_pressure = [&]() __attribute__((always_inline)) {
        // ----------- divergence of v
        for (int i = 0; i < C.ny; ++i) {
            size_t ii = i + halo;
            for (int j = 0; j < C.nx; ++j) {
                size_t jj = j + halo;
                Dtype sumx = 0.0, sumy = 0.0;
                for (int k = 1; k <= halo; ++k) {
                    sumx += coeff[k - 1] * (vx[idx(ii, jj + k)] - vx[idx(ii, jj - k)]);
                    sumy += coeff[k - 1] * (vy[idx(ii + k, jj)] - vy[idx(ii - k, jj)]);
                }
                // Commpiler sometimes works in mysterious ways. Storing in div and then later 
                // updating p is 5Mpts/s faster
                div[idx_s(i, j, C.nx)] = Dtype(sumx * inv_dx + sumy * inv_dy);
            }
        }

        // ------------- update pressure
        for (int i = 0; i < C.ny; ++i) {
            size_t ii = i + halo;
            for (int j = 0; j < C.nx; ++j) {
                size_t jj = j + halo;
                p[idx_s(ii, jj, stride)] -= Dtype(dt) * rhoc2 * div[idx_s(i, j, C.nx)];
            }
        }
    };

    auto update_velosity = [&]() __attribute__((always_inline)) {
        for (int i = 0; i < C.ny; ++i) {
            size_t ii = i + halo;
            for (int j = 0; j < C.nx; ++j) {
                size_t jj = j + halo;
                Dtype gpx = 0.0, gpy = 0.0;
                for (int k = 1; k <= halo; ++k) {
                    gpx += coeff[k - 1] * (p[idx(ii, jj + k)] - p[idx(ii, jj - k)]);
                    gpy += coeff[k - 1] * (p[idx(ii + k, jj)] - p[idx(ii - k, jj)]);
                }
                vx[idx(ii, jj)] -= dt * inv_rho * gpx * inv_dx;
                vy[idx(ii, jj)] -= dt * inv_rho * gpy * inv_dy;
            }
        }
    };

    auto source = [&]() {
        if (C.src_type == "gaussian") {
            return gaussian_wavelet<Dtype>;
        } else {
            return ricker_wavelet<Dtype>;
        }
    }();

    auto zero_normal_v = [&]() {
        for (size_t i = halo; i < Ny - halo; ++i) {
            vx[idx(i, halo)] = 0.0f;           // west
            vx[idx(i, Nx - halo - 1)] = 0.0f;  // east
        }
        for (size_t j = halo; j < Nx - halo; ++j) {
            vy[idx(halo, j)] = 0.0f;           // south
            vy[idx(Ny - halo - 1, j)] = 0.0f;  // north
        }
    };
    auto refresh_neumann = [&](std::vector<Dtype>& f) {
        // West & East
        for (size_t i = halo; i < Ny - halo; ++i) {
            for (int k = 1; k <= halo; ++k) {
                f[idx(i, halo - k)] = f[idx(i, halo)];
                f[idx(i, Nx - halo + k - 1)] = f[idx(i, Nx - halo - 1)];
            }
        }
        // South & North
        for (int k = 1; k <= halo; ++k) {
            for (size_t j = halo; j < Nx - halo; ++j) {
                f[idx(halo - k, j)] = f[idx(halo, j)];
                f[idx(Ny - halo + k - 1, j)] = f[idx(Ny - halo - 1, j)];
            }
        }
    };

    // ---------------------------------------------------------------------
    auto sim_start = std::chrono::high_resolution_clock::now();

    for (int it = 1; it <= C.n_steps; ++it) {
        Dtype t = it * dt;

        // ----------- update pressure
        update_pressure();

        // ----------- point source
        p[src_pos] += source(t, C.f0, C.amp);

        // ----------- grad p  →  update v
        update_velosity();

        // ----------- enforce rigid boundaries & refresh halos
        zero_normal_v();
        refresh_neumann(p);
        refresh_neumann(vx);
        refresh_neumann(vy);

        // ----------- save snapshot
        if (it % C.output_every == 0) {
            for (int i = 0; i < C.ny; ++i) {
                for (int j = 0; j < C.nx; ++j) {
                    frames.push_back(p[idx(i + halo, j + halo)]);
                }
            }
        }
    }
    auto sim_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> sim_duration = sim_end - sim_start;
    double mptss = (static_cast<double>(C.ny * C.nx * C.n_steps) / 1000000.0) / sim_duration.count();  // million points
    fmt::print("Simulation took {:.2f} seconds. Throughput: {:.2f} Mpts/s\n", sim_duration.count(), mptss);

    // ---------------------------------------------------------------------
    // store as NPZ  (shape: n_frames × ny × nx, C-order)
    {
        std::vector<size_t> shape = {
            static_cast<size_t>(n_frames), static_cast<size_t>(C.ny), static_cast<size_t>(C.nx)};
        cnpy::npz_save("wavefield_cpp.npz", "frames", frames.data(), shape, "w");
    }
    fmt::print("Saved {} frames to wavefield_cpp.npz\n", n_frames);
    return 0;
}
