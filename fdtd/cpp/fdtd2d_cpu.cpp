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

inline size_t idx(size_t i, size_t j, size_t stride) { return i * stride + j; }

// -----------------------------------------------------------------------------
// Simulation
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " config.json\n";
        return 1;
    }
    auto C = Config::read_config(argv[1]);
    fmt::print("{}\n", C);

    const int halo = C.m;
    const size_t Ny = C.ny + 2 * halo;
    const size_t Nx = C.nx + 2 * halo;
    const size_t stride = Nx;

    const double dt = 0.4 * std::min(C.dx, C.dy) / (C.c * std::sqrt(2.0));

    std::vector<float> p(Ny * Nx, 0.0f);
    std::vector<float> vx(Ny * Nx, 0.0f);
    std::vector<float> vy(Ny * Nx, 0.0f);

    const int n_frames = C.n_steps / C.output_every + 1;
    std::vector<float> frames;
    frames.reserve(static_cast<size_t>(n_frames) * C.ny * C.nx);

    auto coeff = C.coeffs;  // copy for brevity
    auto refresh = [&](std::vector<float>& f) {
        // West & East
        for (size_t i = halo; i < Ny - halo; ++i) {
            for (int k = 1; k <= halo; ++k) {
                f[idx(i, halo - k, stride)] = f[idx(i, halo, stride)];
                f[idx(i, Nx - halo + k - 1, stride)] = f[idx(i, Nx - halo - 1, stride)];
            }
        }
        // South & North
        for (int k = 1; k <= halo; ++k) {
            for (size_t j = halo; j < Nx - halo; ++j) {
                f[idx(halo - k, j, stride)] = f[idx(halo, j, stride)];
                f[idx(Ny - halo + k - 1, j, stride)] = f[idx(Ny - halo - 1, j, stride)];
            }
        }
    };

    auto zero_normal_v = [&]() {
        for (size_t i = halo; i < Ny - halo; ++i) {
            vx[idx(i, halo, stride)] = 0.0f;           // west
            vx[idx(i, Nx - halo - 1, stride)] = 0.0f;  // east
        }
        for (size_t j = halo; j < Nx - halo; ++j) {
            vy[idx(halo, j, stride)] = 0.0f;           // south
            vy[idx(Ny - halo - 1, j, stride)] = 0.0f;  // north
        }
    };

    // Pre-compute reciprocal distances
    const float inv_dx = 1.0f / C.dx;
    const float inv_dy = 1.0f / C.dy;
    const float rhoc2 = static_cast<float>(C.rho * C.c * C.c);
    const float inv_rho = static_cast<float>(1.0 / C.rho);

    // ---------------------------------------------------------------------
    // output frame 0  (pressure is all zeros)
    for (int i = 0; i < C.ny; ++i) {
        for (int j = 0; j < C.nx; ++j) {
            frames.push_back(p[idx(i + halo, j + halo, stride)]);
        }
    }

    // ---------------------------------------------------------------------
    auto sim_start = std::chrono::high_resolution_clock::now();

    for (int it = 1; it <= C.n_steps; ++it) {
        double t = it * dt;

        // ----------- divergence of v
        std::vector<float> div(C.ny * C.nx, 0.0f);
        for (int i = 0; i < C.ny; ++i) {
            size_t ii = i + halo;
            for (int j = 0; j < C.nx; ++j) {
                size_t jj = j + halo;
                double sumx = 0.0, sumy = 0.0;
                for (int k = 1; k <= halo; ++k) {
                    sumx += coeff[k - 1] * (double(vx[idx(ii, jj + k, stride)]) - double(vx[idx(ii, jj - k, stride)]));
                    sumy += coeff[k - 1] * (double(vy[idx(ii + k, jj, stride)]) - double(vy[idx(ii - k, jj, stride)]));
                }
                div[idx(i, j, C.nx)] = float(sumx * inv_dx + sumy * inv_dy);
            }
        }

        // ----------- update pressure
        for (int i = 0; i < C.ny; ++i) {
            size_t ii = i + halo;
            for (int j = 0; j < C.nx; ++j) {
                size_t jj = j + halo;
                p[idx(ii, jj, stride)] -= float(dt) * rhoc2 * div[idx(i, j, C.nx)];
            }
        }

        // ----------- point source
        if (C.src_type == "gaussian") {
            p[idx(C.sy, C.sx, stride)] += float(gaussian_wavelet(t, C.f0, C.amp));
        } else {
            p[idx(C.sy, C.sx, stride)] += float(ricker_wavelet(t, C.f0, C.amp));
        }

        // ----------- grad p  →  update v
        for (int i = 0; i < C.ny; ++i) {
            size_t ii = i + halo;
            for (int j = 0; j < C.nx; ++j) {
                size_t jj = j + halo;
                double gpx = 0.0, gpy = 0.0;
                for (int k = 1; k <= halo; ++k) {
                    gpx += coeff[k - 1] * (double(p[idx(ii, jj + k, stride)]) - double(p[idx(ii, jj - k, stride)]));
                    gpy += coeff[k - 1] * (double(p[idx(ii + k, jj, stride)]) - double(p[idx(ii - k, jj, stride)]));
                }
                vx[idx(ii, jj, stride)] -= float(dt) * inv_rho * float(gpx * inv_dx);
                vy[idx(ii, jj, stride)] -= float(dt) * inv_rho * float(gpy * inv_dy);
            }
        }

        // ----------- enforce rigid boundaries & refresh halos
        zero_normal_v();
        refresh(p);
        refresh(vx);
        refresh(vy);

        // ----------- save snapshot
        if (it % C.output_every == 0) {
            for (int i = 0; i < C.ny; ++i) {
                for (int j = 0; j < C.nx; ++j) {
                    frames.push_back(p[idx(i + halo, j + halo, stride)]);
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
