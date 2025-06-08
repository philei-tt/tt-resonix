// fdtd2d.cpp  –  single-threaded 2-D acoustic FDTD, C++17
//
// Compile via the accompanying CMakeLists.txt.
// Usage:  ./fdtd2d <config.json>
//
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <cnpy.h>             // third-party/cnpy
#include <nlohmann/json.hpp>  // single-header nlohmann/json

using json = nlohmann::json;
using std::size_t;

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------
template <typename T>
inline T sqr(T x) {
    return x * x;
}

std::vector<double> make_fd_coeffs(int m) {
    // Solve  Σ_k c_k k^(2n+1) = rhs_n,  n = 0 … m-1
    std::vector<double> rhs(m, 0.0);
    rhs[0] = 0.5;

    std::vector<std::vector<double>> A(m, std::vector<double>(m));
    for (int n = 0; n < m; ++n) {
        for (int k = 1; k <= m; ++k) {
            A[n][k - 1] = std::pow(k, 2 * n + 1);
        }
    }

    // Gaussian elimination (tiny m, so simplicity beats libraries)
    for (int i = 0; i < m; ++i) {
        // pivot
        double piv = A[i][i];
        for (int j = i; j < m; ++j) {
            A[i][j] /= piv;
        }
        rhs[i] /= piv;
        // eliminate
        for (int r = 0; r < m; ++r) {
            if (r != i) {
                double f = A[r][i];
                for (int c = i; c < m; ++c) {
                    A[r][c] -= f * A[i][c];
                }
                rhs[r] -= f * rhs[i];
            }
        }
    }
    return rhs;  // length-m
}

struct Config {
    int nx, ny, n_steps, output_every;
    double dx, dy, c, rho;
    std::vector<double> coeffs;  // c1..cm
    int m;                       // halo width
    // source
    std::string src_type;
    double f0, amp;
    int sx, sy;  // in core indices
};

// operator<< for Config:
inline std::ostream& operator<<(std::ostream& os, const Config& cfg) {
    os << "Config(ny=" << cfg.ny << ", nx=" << cfg.nx << ", dx=" << cfg.dx << ", dy=" << cfg.dy << ", c=" << cfg.c
       << ", rho=" << cfg.rho << ", n_steps=" << cfg.n_steps << ", output_every=" << cfg.output_every << ", coeffs=[";
    for (size_t i = 0; i < cfg.coeffs.size(); ++i) {
        os << cfg.coeffs[i];
        if (i < cfg.coeffs.size() - 1) {
            os << ", ";
        }
    }
    os << "], m=" << cfg.m << ", src_type='" << cfg.src_type << "', f0=" << cfg.f0 << ", amp=" << cfg.amp
       << ", sx=" << cfg.sx << ", sy=" << cfg.sy << ")";
    return os;
}

Config read_config(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open config file: " + path);
    }
    json j;
    try {
        // Enable comments during parsing
        j = json::parse(
            file,
            /*cb=*/nullptr,
            /*allow_exceptions=*/true,  // allow exceptions to be thrown
            /*ignore_comments=*/true    // ignore comments in the JSON file
        );
    } catch (const std::exception& e) {
        throw std::runtime_error("Error parsing JSON: " + std::string(e.what()));
    }

    Config cfg;
    cfg.nx = j["nx"];
    cfg.ny = j["ny"];
    cfg.dx = j["dx"];
    cfg.dy = j["dy"];
    cfg.c = j["c"];
    cfg.rho = j["rho"];
    cfg.n_steps = j["n_steps"];
    cfg.output_every = j["output_every"];

    const auto& d = j["derivative"];
    if (d.contains("coeffs")) {
        for (double c : d["coeffs"]) {
            cfg.coeffs.push_back(c);
        }
    } else if (d.contains("m")) {
        cfg.coeffs = make_fd_coeffs(d["m"]);
    } else {
        throw std::runtime_error("derivative needs 'coeffs' or 'm'");
    }
    cfg.m = static_cast<int>(cfg.coeffs.size());

    // source
    const auto& s = j["source"];
    cfg.src_type = s["type"];
    cfg.f0 = s["frequency"];
    cfg.amp = s["amplitude"];
    cfg.sx = static_cast<int>(s["position"][0]) + cfg.m;
    cfg.sy = static_cast<int>(s["position"][1]) + cfg.m;
    return cfg;
}

inline size_t idx(size_t i, size_t j, size_t stride) { return i * stride + j; }

double gaussian_wavelet(double t, double f0, double amp) {
    double tau = 1.0 / f0, t0 = 3.0 * tau;
    return amp * std::exp(-sqr(t - t0) / sqr(tau));
}
double ricker_wavelet(double t, double f0, double amp) {
    double a = M_PI * f0 * (t - 1.0 / f0);
    return amp * (1.0 - 2.0 * sqr(a)) * std::exp(-sqr(a));
}

// -----------------------------------------------------------------------------
// Simulation
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " config.json\n";
        return 1;
    }
    Config C = read_config(argv[1]);
    std::cout << "Config: " << C << "\n";

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

    // ---------------------------------------------------------------------
    // store as NPZ  (shape: n_frames × ny × nx, C-order)
    {
        std::vector<size_t> shape = {
            static_cast<size_t>(n_frames), static_cast<size_t>(C.ny), static_cast<size_t>(C.nx)};
        cnpy::npz_save("wavefield_cpp.npz", "frames", frames.data(), shape, "w");
    }
    std::cout << "Saved " << n_frames << " frames to wavefield_cpp.npz\n";
    return 0;
}
