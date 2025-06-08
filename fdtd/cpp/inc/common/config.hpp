#pragma once

#include <fmt/core.h>
#include <vector>
#include <fstream>
#include <nlohmann/json.hpp>

struct Config {
    int nx, ny, n_steps, output_every;
    double dx, dy, c, rho;
    std::vector<double> coeffs;  // c1..cm
    int m;                       // halo width
    // source
    std::string src_type;
    double f0, amp;
    int sx, sy;  // in core indices

    static Config read_config(const std::string& path);
};

// TODO: Move to a cpp file
// make std::vector<T> printable with fmt:
template <typename T>
struct fmt::formatter<std::vector<T>> : fmt::formatter<std::string> {
    template <typename FormatContext>
    auto format(const std::vector<T>& vec, FormatContext& ctx) const {
        std::string str = "[";
        for (size_t i = 0; i < vec.size(); ++i) {
            str += fmt::to_string(vec[i]);
            if (i < vec.size() - 1) {
                str += ", ";
            }
        }
        str += "]";
        return fmt::formatter<std::string>::format(str, ctx);
    }
};

// TODO: Move to a cpp file
// make Config printable with fmt:
template <>
struct fmt::formatter<Config> : fmt::formatter<std::string> {
    template <typename FormatContext>
    auto format(const Config& cfg, FormatContext& ctx) const {
        std::string str = fmt::format(
            "Config(\n\t ny={}, nx={},\n\t dx={}, dy={},\n\t c={}, rho={},\n\t n_steps={}, output_every={},\n\t m={}, "
            "coeffs={},\n\t "
            "src_type='{}', f0={}, amp={}, sx={}, sy={}\n)",
            cfg.ny,
            cfg.nx,
            cfg.dx,
            cfg.dy,
            cfg.c,
            cfg.rho,
            cfg.n_steps,
            cfg.output_every,
            cfg.m,
            cfg.coeffs,
            cfg.src_type,
            cfg.f0,
            cfg.amp,
            cfg.sx,
            cfg.sy);
        return fmt::formatter<std::string>::format(str, ctx);
    }
};
