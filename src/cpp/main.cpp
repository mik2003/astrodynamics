#define FORCE_IMPORT_ARRAY

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <array>
#include <xtensor-python/pyarray.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/misc/xmanipulation.hpp>
#include <xtensor/views/xview.hpp>

namespace py = pybind11;

// Point-mass N-body derivative function
xt::xarray<double> point_mass_cpp(const xt::pyarray<double>& state,
                                  const xt::pyarray<double>& mu) {
    std::size_t n = mu.size();  // number of bodies
    if (state.size() != 6 * n)
        throw std::runtime_error("State vector size mismatch");

    // Acceleration array
    xt::xarray<double> accel =
        xt::zeros<double>(std::array<std::size_t, 2>{3, n});

    // Compute accelerations
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            auto r = xt::view(state, xt::range(3 * j, 3 * j + 3)) -
                     xt::view(state, xt::range(3 * i, 3 * i + 3));
            double dist = std::sqrt(xt::sum(r * r)());
            xt::view(accel, xt::range(0, 3), i) +=
                mu(j) * r / (dist * dist * dist);
        }
    }

    // Combine velocities and accelerations into derivative
    xt::xarray<double> dstate =
        xt::zeros<double>(std::array<std::size_t, 2>{6 * n});
    xt::view(dstate, xt::range(0, 3 * n)) =
        xt::view(state, xt::range(3 * n, 6 * n));  // velocities
    xt::view(dstate, xt::range(3 * n, 6 * n)) = xt::flatten(accel);

    return dstate;
}

// Pybind11 module
PYBIND11_MODULE(_cpp_force_kernel, m) {
    xt::import_numpy();

    m.doc() =
        "C++ point-mass N-body gravity module (header-only xtensor-python)";
    m.def("point_mass_cpp", &point_mass_cpp, "Compute N-body derivative",
          py::arg("mu"), py::arg("state"));
}
