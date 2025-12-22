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

    // Reshape state to (n,6)
    auto state_reshaped =
        xt::reshape_view(state, std::array<std::size_t, 2>{n, 6});
    auto pos = xt::view(state_reshaped, xt::all(), xt::range(0, 3));
    auto vel = xt::view(state_reshaped, xt::all(), xt::range(3, 6));

    // Compute all pairwise relative positions: (n,3,n)
    auto r_ij =
        xt::expand_dims(pos, 2) - xt::expand_dims(pos, 0);  // shape (n,3,n)

    // Compute squared distances, avoid self-interaction
    xt::xarray<double> dist_sq = xt::sum(r_ij * r_ij, {1});  // shape (n,n)
    for (std::size_t i = 0; i < n; ++i) {
        dist_sq(i, i) = std::numeric_limits<double>::infinity();
    }

    // Compute 1 / r^3
    xt::xarray<double> inv_dist3 =
        1.0 / (dist_sq * xt::sqrt(dist_sq));  // shape (n,n)

    // Multiply by mu_j (broadcasting)
    auto mu_reshaped =
        xt::reshape_view(mu, std::array<std::size_t, 3>{1, 1, n});
    auto inv_dist3_reshaped =
        xt::reshape_view(inv_dist3, std::array<std::size_t, 3>{n, 1, n});
    xt::xarray<double> accel_contrib = r_ij * mu_reshaped * inv_dist3_reshaped;

    // Sum contributions along axis=2 to get (n,3) accelerations
    xt::xarray<double> accel = xt::sum(accel_contrib, 2);

    // Combine velocities and accelerations into dstate
    xt::xarray<double> dstate = xt::zeros<double>({6 * n});
    xt::view(dstate, xt::range(0, 3 * n)) = xt::flatten(accel);
    xt::view(dstate, xt::range(3 * n, 6 * n)) = xt::flatten(vel);

    return dstate;
}

// Pybind11 module
PYBIND11_MODULE(_cpp_force_kernel, m) {
    xt::import_numpy();

    m.doc() = "C++ point-mass N-body gravity module";
    m.def("point_mass_cpp", &point_mass_cpp, "Compute N-body derivative",
          py::arg("state"), py::arg("mu"));
}
