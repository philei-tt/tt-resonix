// fdtd2d_cpu_mt.cpp  –  multi-threaded 2-D acoustic FDTD, C++17
//
// Threads are split along the Y axis: each one processes an independent
// band of rows and the entire X span.
//
// Usage:  ./fdtd2d_mt <config.json>
//
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <fmt/core.h>

#include <cnpy.h>             // third-party/cnpy
#include <nlohmann/json.hpp>  // single-header nlohmann/json

#include "common/config.hpp"
#include "common/sim.hpp"

using std::size_t;
using Dtype = float;  // simulation data type  --------------------------------

// -----------------------------------------------------------------------------
// A tiny, reusable barrier for C++17
// -----------------------------------------------------------------------------
class Barrier {
public:
    explicit Barrier(size_t n) : thresh_(n), count_(n), gen_(0) {}
    void wait() {
        std::unique_lock<std::mutex> lk(m_);
        const auto g = gen_;
        if (--count_ == 0) {
            ++gen_;
            count_ = thresh_;
            lk.unlock();
            cv_.notify_all();
        } else {
            cv_.wait(lk, [&] { return g != gen_; });
        }
    }

private:
    std::mutex m_;
    std::condition_variable cv_;
    size_t thresh_, count_, gen_;
};

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

    // ---------------------------------------------------------------------
    // Geometry & constants
    // ---------------------------------------------------------------------
    const int halo = C.m;  // stencil half-width
    const size_t Ny = C.ny + 2 * halo;
    const size_t Nx = C.nx + 2 * halo;
    const size_t stride = Nx;  // row stride

    const Dtype dt = 0.4f * std::min(C.dx, C.dy) / (C.c * std::sqrt(2.0f));

    std::vector<Dtype> p(Ny * Nx, 0.0f);
    std::vector<Dtype> vx(Ny * Nx, 0.0f);
    std::vector<Dtype> vy(Ny * Nx, 0.0f);
    std::vector<Dtype> div(C.ny * C.nx, 0.0f);

    // fast index helpers (C-order)
    auto idx_s = [&](size_t i, size_t j, const size_t s) { return i * s + j; };
    auto idx = [&](size_t i, size_t j) { return i * stride + j; };

    const size_t src_pos = idx(C.sy, C.sx);

    const int n_frames = C.output_every == 0 ? 0 : C.n_steps / C.output_every + 1;
    std::vector<Dtype> frames;
    frames.reserve(static_cast<size_t>(n_frames) * C.ny * C.nx);

    const auto coeff = C.coeffs;
    const Dtype inv_dx = 1.0f / C.dx;
    const Dtype inv_dy = 1.0f / C.dy;
    const Dtype rhoc2 = static_cast<Dtype>(C.rho * C.c * C.c);
    const Dtype inv_rho = static_cast<Dtype>(1.0 / C.rho);

    // ---------------------------------------------------------------------
    // Helpers that touch *whole* domain – executed by thread 0 only
    // ---------------------------------------------------------------------
    auto zero_normal_v = [&] {
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
        // West / East
        for (size_t i = halo; i < Ny - halo; ++i) {
            for (int k = 1; k <= halo; ++k) {
                f[idx(i, halo - k)] = f[idx(i, halo)];
                f[idx(i, Nx - halo + k - 1)] = f[idx(i, Nx - halo - 1)];
            }
        }
        // South / North
        for (int k = 1; k <= halo; ++k) {
            for (size_t j = halo; j < Nx - halo; ++j) {
                f[idx(halo - k, j)] = f[idx(halo, j)];
                f[idx(Ny - halo + k - 1, j)] = f[idx(Ny - halo - 1, j)];
            }
        }
    };
    auto source = [&]() { return (C.src_type == "gaussian") ? gaussian_wavelet<Dtype> : ricker_wavelet<Dtype>; }();
    // pre-compute full frame 0 on the calling thread
    if (n_frames > 0) {
        for (int i = 0; i < C.ny; ++i) {
            for (int j = 0; j < C.nx; ++j) {
                frames.push_back(p[idx(i + halo, j + halo)]);
            }
        }
    }

    // ---------------------------------------------------------------------
    // THREAD SET-UP
    // ---------------------------------------------------------------------
    const auto n_threads = std::thread::hardware_concurrency();
    Barrier barrier(n_threads);

    const int ny_int = C.ny;
    const int chunk = ny_int / n_threads;
    const int rem = ny_int % n_threads;

    const double total_pts = static_cast<double>(C.ny) * C.nx * C.n_steps;

    // ---------------------------------------------------------------------
    auto worker = [&](unsigned tid) {
        // -------- slice this thread owns along Y --------------------------
        const int extra = (int)tid < rem ? 1 : 0;
        const int i0 = tid * chunk + std::min<int>(tid, rem);
        const int i1 = i0 + chunk + extra;

        // -------- local kernels ------------------------------------------
        auto update_pressure_slice = [&] {
            for (int i = i0; i < i1; ++i) {
                size_t ii = i + halo;
                for (int j = 0; j < C.nx; ++j) {
                    size_t jj = j + halo;
                    Dtype sumx = 0.0f, sumy = 0.0f;
                    for (int k = 1; k <= halo; ++k) {
                        sumx += coeff[k - 1] * (vx[idx(ii, jj + k)] - vx[idx(ii, jj - k)]);
                        sumy += coeff[k - 1] * (vy[idx(ii + k, jj)] - vy[idx(ii - k, jj)]);
                    }
                    div[idx_s(i, j, C.nx)] = static_cast<Dtype>(sumx * inv_dx + sumy * inv_dy);
                }
            }
            for (int i = i0; i < i1; ++i) {
                size_t ii = i + halo;
                for (int j = 0; j < C.nx; ++j) {
                    size_t jj = j + halo;
                    p[idx_s(ii, jj, stride)] -= dt * rhoc2 * div[idx_s(i, j, C.nx)];
                }
            }
        };
        auto update_velocity_slice = [&] {
            for (int i = i0; i < i1; ++i) {
                size_t ii = i + halo;
                for (int j = 0; j < C.nx; ++j) {
                    size_t jj = j + halo;
                    Dtype gpx = 0.0f, gpy = 0.0f;
                    for (int k = 1; k <= halo; ++k) {
                        gpx += coeff[k - 1] * (p[idx(ii, jj + k)] - p[idx(ii, jj - k)]);
                        gpy += coeff[k - 1] * (p[idx(ii + k, jj)] - p[idx(ii - k, jj)]);
                    }
                    vx[idx(ii, jj)] -= dt * inv_rho * gpx * inv_dx;
                    vy[idx(ii, jj)] -= dt * inv_rho * gpy * inv_dy;
                }
            }
        };

        // ------------------- time loop -----------------------------------
        for (int it = 1; it <= C.n_steps; ++it) {
            const Dtype t = it * dt;

            // ----------- update pressure
            update_pressure_slice();
            barrier.wait();

            // ----------- point source
            if (tid == 0) {
                p[src_pos] += source(t, C.f0, C.amp);
            }
            barrier.wait();

            // ----------- grad p  →  update v
            update_velocity_slice();
            barrier.wait();

            // ----------- enforce rigid boundaries & refresh halos
            if (tid == 0) {
                zero_normal_v();
                refresh_neumann(p);
                refresh_neumann(vx);
                refresh_neumann(vy);

                if (C.output_every and it % C.output_every == 0) {
                    for (int i = 0; i < C.ny; ++i) {
                        for (int j = 0; j < C.nx; ++j) {
                            frames.push_back(p[idx(i + halo, j + halo)]);
                        }
                    }
                }
            }
            barrier.wait();
        }
    };

    // ---------------------- run ------------------------------------------
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> pool;
    pool.reserve(n_threads);
    for (unsigned t = 0; t < n_threads; ++t) {
        pool.emplace_back(worker, t);
    }
    for (auto& th : pool) {
        th.join();
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();
    const double mpts_s = (total_pts / 1.0e6) / secs;

    fmt::print("Simulation took {:.2f} s – {:.2f} Mpts/s on {} threads\n", secs, mpts_s, n_threads);

    // ---------------------------------------------------------------------
    // store as NPZ  (shape: n_frames × ny × nx, C-order)
    {
        std::vector<size_t> shp = {static_cast<size_t>(n_frames), static_cast<size_t>(C.ny), static_cast<size_t>(C.nx)};
        cnpy::npz_save("wavefield_cpp.npz", "frames", frames.data(), shp, "w");
    }
    fmt::print("Saved {} frames to wavefield_cpp.npz\n", n_frames);
    return 0;
}
