// fdtd2d_cuda.cu  –  2‑D acoustic FDTD (CUDA, single‑GPU, C++17)
// -----------------------------------------------------------------------------
// This is a straightforward GPU port of the original fdtd2d_cpu.cpp implementation.
//  * Pressure and velocity fields live exclusively on the GPU during the run – no
//    per‑step host/device shuttling.
//  * Divergence / gradient are computed in‑kernel, eliminating the temporary `div`
//    array used on CPU.
//  * Halo refresh and rigid boundary enforcement are also done on‑device.
//  * Snapshots are copied back only when requested.
// -----------------------------------------------------------------------------
// Build (example CMake stanza):
// add_executable(fdtd2d_cuda fdtd2d_cuda.cu)
// set_target_properties(fdtd2d_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
// target_link_libraries(fdtd2d_cuda PRIVATE fmt::fmt cnpy nlohmann_json::nlohmann_json)
// -----------------------------------------------------------------------------

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>
#include <chrono>
#include <fmt/core.h>

#include <cnpy.h>
#include <nlohmann/json.hpp>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "common/config.hpp"
#include "common/sim.hpp"

using std::size_t;

using Dtype = float;  // simulation data type (32‑bit floats for best GPU perf)
constexpr int MAX_M = 8;  // max supported stencil half‑width (constant memory)

// -----------------------------------------------------------------------------
// Error handling helpers
// -----------------------------------------------------------------------------
#define CUDA_CHECK(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fmt::print(stderr, "CUDA error {} ({}:{})\n", cudaGetErrorString(code), file,
                   line);
        std::exit(code);
    }
}

// -----------------------------------------------------------------------------
// Constant memory for stencil coefficients ( <= MAX_M )
// -----------------------------------------------------------------------------
__constant__ Dtype d_coeff[MAX_M];

// -----------------------------------------------------------------------------
// Kernels
// -----------------------------------------------------------------------------
__global__ void update_pressure_kernel(Dtype * __restrict__ p,
                                       const Dtype * __restrict__ vx,
                                       const Dtype * __restrict__ vy,
                                       Dtype inv_dx, Dtype inv_dy,
                                       Dtype rhoc2, Dtype dt,
                                       int halo, int nx, int ny, int stride) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;  //   0 … nx‑1  (physical)
    int i = blockIdx.y * blockDim.y + threadIdx.y;  //   0 … ny‑1
    if (j >= nx || i >= ny) return;

    int ii = i + halo;  // indices in padded array
    int jj = j + halo;

    Dtype sumx = 0.0f, sumy = 0.0f;
    #pragma unroll
    for (int k = 1; k <= MAX_M; ++k) {
        if (k > halo) break;  // guard for small m
        Dtype c = d_coeff[k - 1];
        sumx += c * (vx[ii * stride + (jj + k)] - vx[ii * stride + (jj - k)]);
        sumy += c * (vy[(ii + k) * stride + jj] - vy[(ii - k) * stride + jj]);
    }

    Dtype div = sumx * inv_dx + sumy * inv_dy;
    p[ii * stride + jj] -= dt * rhoc2 * div;
}

__global__ void update_velocity_kernel(Dtype * __restrict__ vx,
                                       Dtype * __restrict__ vy,
                                       const Dtype * __restrict__ p,
                                       Dtype inv_dx, Dtype inv_dy,
                                       Dtype inv_rho, Dtype dt,
                                       int halo, int nx, int ny, int stride) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= nx || i >= ny) return;

    int ii = i + halo;
    int jj = j + halo;

    Dtype gpx = 0.0f, gpy = 0.0f;
    #pragma unroll
    for (int k = 1; k <= MAX_M; ++k) {
        if (k > halo) break;
        Dtype c = d_coeff[k - 1];
        gpx += c * (p[ii * stride + (jj + k)] - p[ii * stride + (jj - k)]);
        gpy += c * (p[(ii + k) * stride + jj] - p[(ii - k) * stride + jj]);
    }
    vx[ii * stride + jj] -= dt * inv_rho * gpx * inv_dx;
    vy[ii * stride + jj] -= dt * inv_rho * gpy * inv_dy;
}

// Set normal velocity = 0 on rigid boundaries (four edges)
__global__ void zero_normal_v_kernel(Dtype *vx, Dtype *vy,
                                     int halo, int nx, int ny, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Vertical edges (vary i)
    if (tid < ny) {
        int ii = tid + halo;
        vx[ii * stride + halo] = 0.0f;                 // west
        vx[ii * stride + (nx + halo - 1)] = 0.0f;      // east
    }
    // Horizontal edges (reuse tid as j)
    if (tid < nx) {
        int jj = tid + halo;
        vy[halo * stride + jj] = 0.0f;                 // south
        vy[(ny + halo - 1) * stride + jj] = 0.0f;      // north
    }
}

// First‑order (Neumann) halo refresh for a field f (p, vx or vy)
__global__ void refresh_neumann_kernel(Dtype *f, int halo, int nx, int ny, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // replicate columns (west/east)
    if (tid < ny) {
        int ii = tid + halo;
        for (int k = 1; k <= halo; ++k) {
            f[ii * stride + (halo - k)]           = f[ii * stride + halo];               // west halo
            f[ii * stride + (nx + halo - 1 + k)] = f[ii * stride + (nx + halo - 1)];    // east halo
        }
    }
    // replicate rows (south/north)
    if (tid < nx) {
        int jj = tid + halo;
        for (int k = 1; k <= halo; ++k) {
            f[(halo - k) * stride + jj]           = f[halo * stride + jj];               // south halo
            f[(ny + halo - 1 + k) * stride + jj]  = f[(ny + halo - 1) * stride + jj];    // north halo
        }
    }
}

// /* -------- 2. race-free halo refresh  -------- */
// __global__ void refresh_cols(Dtype *f, int halo, int nx, int ny, int stride)
// {
//     int ii = blockIdx.x * blockDim.x + threadIdx.x;          // each thread = one row
//     if (ii >= ny) return;
//     ii += halo;                                              // shift into padded array
//     for (int k = 1; k <= halo; ++k) {
//         f[ii * stride + (halo - k)]            = f[ii * stride + halo];            // west
//         f[ii * stride + (nx + halo - 1 + k)]  = f[ii * stride + (nx + halo - 1)]; // east
//     }
// }

// __global__ void refresh_rows(Dtype *f, int halo, int nx, int ny, int stride)
// {
//     int jj = blockIdx.x * blockDim.x + threadIdx.x;          // each thread = one col
//     if (jj >= nx) return;
//     jj += halo;
//     for (int k = 1; k <= halo; ++k) {
//         f[(halo - k)           * stride + jj] = f[halo * stride + jj];             // south
//         f[(ny + halo - 1 + k)  * stride + jj] = f[(ny + halo - 1) * stride + jj];  // north
//     }
// }


// Dump interior field into frames_d[frame_idx]
__global__ void store_frame_kernel(const Dtype * __restrict__ p,
                                   Dtype * __restrict__ frames,
                                   int frame_idx,
                                   int halo, int nx, int ny, int stride) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= nx || i >= ny) return;

    frames[static_cast<size_t>(frame_idx) * nx * ny + i * nx + j] =
        p[(i + halo) * stride + (j + halo)];
}

// Simple kernel to add a point source at a single index
__global__ void add_source_kernel(Dtype *p, int src_idx, Dtype src_val) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        p[src_idx] += src_val;
    }
}

// -----------------------------------------------------------------------------
int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " config.json\n";
        return 1;
    }

    auto C = Config<Dtype>::read_config(argv[1]);
    fmt::print("{}\n", C);

    const int halo = C.m;
    if (halo > MAX_M) {
        fmt::print(stderr, "ERROR: halo m={} exceeds MAX_M={} (recompile with larger MAX_M)\n",
                   halo, MAX_M);
        return 1;
    }

    const size_t Ny = C.ny + 2 * halo;
    const size_t Nx = C.nx + 2 * halo;
    const size_t stride = Nx;  // contiguous rows

    const Dtype dt = 0.4f * std::min(C.dx, C.dy) / static_cast<Dtype>(C.c * std::sqrt(2.0));

    // ------------------------------------------------------------------
    // Allocate device memory
    // ------------------------------------------------------------------
    Dtype *d_p = nullptr, *d_vx = nullptr, *d_vy = nullptr, *d_frames = nullptr;
    CUDA_CHECK(cudaMalloc(&d_p, Ny * Nx * sizeof(Dtype)));
    CUDA_CHECK(cudaMalloc(&d_vx, Ny * Nx * sizeof(Dtype)));
    CUDA_CHECK(cudaMalloc(&d_vy, Ny * Nx * sizeof(Dtype)));

    CUDA_CHECK(cudaMemset(d_p,  0, Ny * Nx * sizeof(Dtype)));
    CUDA_CHECK(cudaMemset(d_vx, 0, Ny * Nx * sizeof(Dtype)));
    CUDA_CHECK(cudaMemset(d_vy, 0, Ny * Nx * sizeof(Dtype)));

    // Copy stencil coefficients to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(d_coeff, C.coeffs.data(), halo * sizeof(Dtype)));

    // Frames
    const int n_frames = C.output_every == 0 ? 0 : C.n_steps / C.output_every + 1;
    if (n_frames > 0) {
        CUDA_CHECK(cudaMalloc(&d_frames, static_cast<size_t>(n_frames) * C.ny * C.nx * sizeof(Dtype)));
        CUDA_CHECK(cudaMemset(d_frames, 0, static_cast<size_t>(n_frames) * C.ny * C.nx * sizeof(Dtype)));
    }

    // Source index in padded array
    const int src_pos = C.sy * stride + C.sx;

    const Dtype inv_dx = 1.0f / C.dx;
    const Dtype inv_dy = 1.0f / C.dy;
    const Dtype rhoc2  = static_cast<Dtype>(C.rho * C.c * C.c);
    const Dtype inv_rho = static_cast<Dtype>(1.0 / C.rho);

    // ------------------------------------------------------------------
    // Select source wavelet (host side scalar function)
    // ------------------------------------------------------------------
    auto src_func = [&]() {
        if (C.src_type == "gaussian")
            return gaussian_wavelet<Dtype>;
        else
            return ricker_wavelet<Dtype>;
    }();

    // ------------------------------------------------------------------
    // Determine launch parameters
    // ------------------------------------------------------------------
    dim3 block(16, 16);
    dim3 grid((C.nx + block.x - 1) / block.x, (C.ny + block.y - 1) / block.y);
    int threads1D = 256;
    int blocksYX = (std::max(C.nx, C.ny) + threads1D - 1) / threads1D;
    int blocks_rows = (C.ny + threads1D - 1) / threads1D;
    int blocks_cols = (C.nx + threads1D - 1) / threads1D;


    // ------------------------------------------------------------------
    // Output initial (zero) frame #0, already on device → store_frame_kernel
    // ------------------------------------------------------------------
    if (n_frames > 0) {
        store_frame_kernel<<<grid, block>>>(d_p, d_frames, 0, halo, C.nx, C.ny, stride);
        CUDA_CHECK(cudaGetLastError());
    }

    // ------------------------------------------------------------------
    // Timing (CUDA events give device‑side elapsed time only)
    // ------------------------------------------------------------------
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaEventRecord(ev_start));

    // ------------------------------------------------------------------
    // Main time‑stepping loop
    // ------------------------------------------------------------------
    for (int it = 1, frame_id = 1; it <= C.n_steps; ++it) {
        // 1. ∇·v → p
        update_pressure_kernel<<<grid, block>>>(d_p, d_vx, d_vy, inv_dx, inv_dy,
                                                rhoc2, dt, halo, C.nx, C.ny, stride);

        // 2. Add point source (single‑thread kernel)
        Dtype src_val = src_func(it * dt, C.f0, C.amp);
        add_source_kernel<<<1, 1>>>(d_p, src_pos, src_val);

        // 3. ∇p → v
        update_velocity_kernel<<<grid, block>>>(d_vx, d_vy, d_p, inv_dx, inv_dy,
                                                inv_rho, dt, halo, C.nx, C.ny, stride);

        // 4. Rigid boundaries (normal velocity = 0)
        zero_normal_v_kernel<<<blocksYX, threads1D>>>(d_vx, d_vy, halo, C.nx, C.ny, stride);

        // 5. Neumann halo refresh for p, vx, vy
        refresh_neumann_kernel<<<blocksYX, threads1D>>>(d_p,  halo, C.nx, C.ny, stride);
        refresh_neumann_kernel<<<blocksYX, threads1D>>>(d_vx, halo, C.nx, C.ny, stride);
        refresh_neumann_kernel<<<blocksYX, threads1D>>>(d_vy, halo, C.nx, C.ny, stride);
        // refresh_cols<<<blocks_rows, threads1D>>>(d_p,  halo, C.nx, C.ny, stride);
        // refresh_cols<<<blocks_rows, threads1D>>>(d_vx, halo, C.nx, C.ny, stride);
        // refresh_cols<<<blocks_rows, threads1D>>>(d_vy, halo, C.nx, C.ny, stride);

        // refresh_rows<<<blocks_cols, threads1D>>>(d_p,  halo, C.nx, C.ny, stride);
        // refresh_rows<<<blocks_cols, threads1D>>>(d_vx, halo, C.nx, C.ny, stride);
        // refresh_rows<<<blocks_cols, threads1D>>>(d_vy, halo, C.nx, C.ny, stride);


        // 6. Snapshot
        if (C.output_every && (it % C.output_every == 0)) {
            if (n_frames > 0) {
                store_frame_kernel<<<grid, block>>>(d_p, d_frames, frame_id++,
                                                    halo, C.nx, C.ny, stride);
            }
        }
    }

    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float sim_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&sim_ms, ev_start, ev_stop));
    double sim_seconds = sim_ms * 1e-3;

    double mptss =
        (static_cast<double>(C.ny) * C.nx * C.n_steps / 1e6) / sim_seconds;
    fmt::print("Simulation took {:.2f} seconds (GPU). Throughput: {:.2f} Mpts/s\n",
               sim_seconds, mptss);

    // ------------------------------------------------------------------
    // Copy frames back to host and save NPZ
    // ------------------------------------------------------------------
    std::vector<Dtype> frames;
    frames.resize(static_cast<size_t>(n_frames) * C.ny * C.nx);
    if (n_frames > 0) {
        CUDA_CHECK(cudaMemcpy(frames.data(), d_frames,
                              frames.size() * sizeof(Dtype), cudaMemcpyDeviceToHost));
        std::vector<size_t> shape = {static_cast<size_t>(n_frames),
                                     static_cast<size_t>(C.ny),
                                     static_cast<size_t>(C.nx)};
        cnpy::npz_save("wavefield_cuda.npz", "frames", frames.data(), shape, "w");
        fmt::print("Saved {} frames to wavefield_cuda.npz\n", n_frames);
    }

    // ------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------
    cudaFree(d_p);
    cudaFree(d_vx);
    cudaFree(d_vy);
    cudaFree(d_frames);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    return 0;
}
