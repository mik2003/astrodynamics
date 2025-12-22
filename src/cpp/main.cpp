#define FORCE_IMPORT_ARRAY

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <array>
#include <xtensor-python/pyarray.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/views/xview.hpp>

namespace py = pybind11;

xt::xarray<double> point_mass_cpp(const xt::pyarray<double>& state,
                                  const xt::pyarray<double>& mu) {
    std::size_t n = mu.size();
    if (state.size() != 6 * n)
        throw std::runtime_error("State vector size mismatch");

    // Separate positions and velocities
    const std::size_t shape[2] = {n, 6};
    auto state_reshaped = xt::reshape_view(state, shape);  // safe

    auto pos = xt::view(state_reshaped, xt::all(), xt::range(0, 3));
    auto vel = xt::view(state_reshaped, xt::all(), xt::range(3, 6));

    // Initialize accelerations (n,3)
    xt::xarray<double> accel =
        xt::zeros<double>(std::array<std::size_t, 2>{n, 3});

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            auto r = xt::view(pos, j, xt::all()) - xt::view(pos, i, xt::all());
            double dist = std::sqrt(xt::sum(r * r)());
            xt::view(accel, i, xt::all()) += mu(j) * r / (dist * dist * dist);
        }
    }

    // Flatten accelerations and combine with velocities
    xt::xarray<double> dstate(6 * n);
    xt::view(dstate, xt::range(0, 3 * n)) = xt::flatten(accel);
    xt::view(dstate, xt::range(3 * n, 6 * n)) = xt::flatten(vel);

    return dstate;
}

// Pybind11 module
PYBIND11_MODULE(_cpp_force_kernel, m) {
    xt::import_numpy();

    m.doc() =
        "C++ point-mass N-body gravity module (header-only xtensor-python)";
    m.def("point_mass_cpp", &point_mass_cpp, "Compute N-body derivative",
          py::arg("state"), py::arg("mu"));
}
