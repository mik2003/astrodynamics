#define FORCE_IMPORT_ARRAY

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cmath>
#include <vector>

namespace py = pybind11;

// Fast version using raw arrays
py::array_t<double> point_mass_cpp_fast(py::array_t<double> state_py,
                                        py::array_t<double> mu_py) {
    // Get buffer info
    auto state_buf = state_py.request();
    auto mu_buf = mu_py.request();

    if (state_buf.ndim != 1 || mu_buf.ndim != 1)
        throw std::runtime_error("Input must be 1D arrays");

    double* state = static_cast<double*>(state_buf.ptr);
    double* mu = static_cast<double*>(mu_buf.ptr);

    size_t n = mu_buf.size;
    size_t state_size = state_buf.size;

    if (state_size != 6 * n)
        throw std::runtime_error("State vector size mismatch");

    // Create output array
    py::array_t<double> result = py::array_t<double>(state_size);
    auto result_buf = result.request();
    double* dstate = static_cast<double*>(result_buf.ptr);

    // Copy velocities to output (first half of dstate)
    for (size_t i = 0; i < 3 * n; ++i) {
        dstate[i] = state[i + 3 * n];  // velocities
    }

    // Compute accelerations
    for (size_t i = 0; i < n; ++i) {
        double ax = 0.0, ay = 0.0, az = 0.0;

        double xi = state[6 * i];
        double yi = state[6 * i + 1];
        double zi = state[6 * i + 2];

        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;

            double dx = xi - state[6 * j];
            double dy = yi - state[6 * j + 1];
            double dz = zi - state[6 * j + 2];

            double r2 = dx * dx + dy * dy + dz * dz;
            double inv_r3 = 1.0 / (r2 * std::sqrt(r2));
            double factor = mu[j] * inv_r3;

            ax += dx * factor;
            ay += dy * factor;
            az += dz * factor;
        }

        dstate[3 * n + 3 * i] = ax;
        dstate[3 * n + 3 * i + 1] = ay;
        dstate[3 * n + 3 * i + 2] = az;
    }

    return result;
}

// Even faster version with SIMD hints and optimized memory access
py::array_t<double> point_mass_cpp_optimized(py::array_t<double> state_py,
                                             py::array_t<double> mu_py) {
    // Get buffer info
    auto state_buf = state_py.request();
    auto mu_buf = mu_py.request();

    double* state = static_cast<double*>(state_buf.ptr);
    double* mu = static_cast<double*>(mu_buf.ptr);

    size_t n = mu_buf.size;
    size_t total = state_buf.size;

    // Create output array
    py::array_t<double> result = py::array_t<double>(total);
    double* dstate = static_cast<double*>(result.request().ptr);

    // Copy velocities (first half)
    size_t vel_offset = 3 * n;
    for (size_t i = 0; i < vel_offset; ++i) {
        dstate[i] = state[i + vel_offset];
    }

    // Pre-cache positions for better memory access
    std::vector<double> pos_x(n), pos_y(n), pos_z(n);
    for (size_t i = 0; i < n; ++i) {
        size_t base = 6 * i;
        pos_x[i] = state[base];
        pos_y[i] = state[base + 1];
        pos_z[i] = state[base + 2];
    }

// Compute accelerations
#pragma omp parallel for if (n > 100)  // Enable OpenMP for large n
    for (size_t i = 0; i < n; ++i) {
        double xi = pos_x[i];
        double yi = pos_y[i];
        double zi = pos_z[i];

        double ax = 0.0, ay = 0.0, az = 0.0;

        // Unroll small loops for better performance
        if (n < 8) {
            // Small n: simple loop
            for (size_t j = 0; j < n; ++j) {
                if (i == j) continue;

                double dx = xi - pos_x[j];
                double dy = yi - pos_y[j];
                double dz = zi - pos_z[j];

                double r2 = dx * dx + dy * dy + dz * dz;
                double inv_r3 = 1.0 / (r2 * std::sqrt(r2));
                double factor = mu[j] * inv_r3;

                ax += dx * factor;
                ay += dy * factor;
                az += dz * factor;
            }
        } else {
            // Larger n: optimized loop
            for (size_t j = 0; j < n; ++j) {
                if (i == j) continue;

                double dx = xi - pos_x[j];
                double dy = yi - pos_y[j];
                double dz = zi - pos_z[j];

                double r2 = dx * dx + dy * dy + dz * dz;
                // Fast inverse sqrt approximation (optional)
                double inv_r = 1.0 / std::sqrt(r2);
                double inv_r3 = inv_r * inv_r * inv_r;
                double factor = mu[j] * inv_r3;

                ax += dx * factor;
                ay += dy * factor;
                az += dz * factor;
            }
        }

        size_t acc_offset = vel_offset + 3 * i;
        dstate[acc_offset] = ax;
        dstate[acc_offset + 1] = ay;
        dstate[acc_offset + 2] = az;
    }

    return result;
}

PYBIND11_MODULE(_cpp_force_kernel, m) {
    m.doc() = "C++ point-mass N-body gravity module";

    m.def("point_mass_cpp", &point_mass_cpp_fast,
          "Compute N-body derivative (fast)", py::arg("state"), py::arg("mu"));

    m.def("point_mass_cpp_optimized", &point_mass_cpp_optimized,
          "Compute N-body derivative (optimized)", py::arg("state"),
          py::arg("mu"));
}