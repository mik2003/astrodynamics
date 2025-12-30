#define FORCE_IMPORT_ARRAY

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cmath>
#include <cstddef>

namespace py = pybind11;

/* =========================
   Fast symmetric force kernel
   ========================= */

inline void point_mass_force_kernel(
    const double* __restrict__ state,  // size: 6*n
    size_t n,
    const double* __restrict__ mu,  // size: n
    double* __restrict__ out        // size: 6*n
) {
    const size_t vel_offset = 3 * n;

    // r' = v
    for (size_t k = 0; k < vel_offset; ++k) {
        out[k] = state[k + vel_offset];
    }

    // zero accelerations
    for (size_t k = 0; k < vel_offset; ++k) {
        out[k + vel_offset] = 0.0;
    }

    // symmetric gravity
    for (size_t i = 0; i < n; ++i) {
        const double xi = state[3 * i];
        const double yi = state[3 * i + 1];
        const double zi = state[3 * i + 2];

        const double mi = mu[i];

        for (size_t j = i + 1; j < n; ++j) {
            const double dx = xi - state[3 * j];
            const double dy = yi - state[3 * j + 1];
            const double dz = zi - state[3 * j + 2];

            const double r2 = dx * dx + dy * dy + dz * dz;
            const double inv_r = 1.0 / std::sqrt(r2);
            const double inv_r3 = inv_r * inv_r * inv_r;

            const double fx = dx * inv_r3;
            const double fy = dy * inv_r3;
            const double fz = dz * inv_r3;

            const double mj = mu[j];

            out[vel_offset + 3 * i] -= mj * fx;
            out[vel_offset + 3 * i + 1] -= mj * fy;
            out[vel_offset + 3 * i + 2] -= mj * fz;

            out[vel_offset + 3 * j] += mi * fx;
            out[vel_offset + 3 * j + 1] += mi * fy;
            out[vel_offset + 3 * j + 2] += mi * fz;
        }
    }
}

/* =========================
   Python-facing wrapper
   ========================= */

void point_mass_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> state,
    py::array_t<double, py::array::c_style | py::array::forcecast> mu,
    py::array_t<double, py::array::c_style | py::array::forcecast> out) {
    auto state_buf = state.request();
    auto mu_buf = mu.request();
    auto out_buf = out.request();

    if (state_buf.ndim != 1 || mu_buf.ndim != 1 || out_buf.ndim != 1) {
        throw std::runtime_error("All arrays must be 1D");
    }

    const size_t n = mu_buf.size;
    if (state_buf.size != 6 * n || out_buf.size != 6 * n) {
        throw std::runtime_error("state and out must have size 6*n");
    }

    const double* s = static_cast<const double*>(state_buf.ptr);
    const double* m = static_cast<const double*>(mu_buf.ptr);
    double* o = static_cast<double*>(out_buf.ptr);

    point_mass_force_kernel(s, n, m, o);
}

/* =========================
   Module definition
   ========================= */

PYBIND11_MODULE(_cpp_force_kernel, m) {
    m.doc() = "C++ point-mass N-body gravity module (symmetric, fast)";

    m.def("point_mass_cpp", &point_mass_cpp, py::arg("state"), py::arg("mu"),
          py::arg("out"));
}
