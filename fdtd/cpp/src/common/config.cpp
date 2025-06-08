#include "common/config.hpp"
#include "common/sim.hpp"

using nlohmann::json;

template<typename Dtype>
Config<Dtype> Config<Dtype>::read_config(const std::string& path) {
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
        cfg.coeffs = make_fd_coeffs<Dtype>(d["m"]);
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


template Config<float> Config<float>::read_config(const std::string& path);
template Config<double> Config<double>::read_config(const std::string& path);