#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/core/xmath.hpp>

namespace py = pybind11;

// Simple function to test the build
xt::xarray<double> add_arrays(const xt::xarray<double>& a,
                              const xt::xarray<double>& b) {
    return a + b;
}

// Process Python list - FIXED
double sum_list(py::list py_list) {
    std::vector<double> vec;
    for (auto item : py_list) {
        vec.push_back(item.cast<double>());
    }

    // Use xt::adapt from xtensor/xadapt.hpp
    auto arr = xt::adapt(vec, {vec.size()});
    return xt::sum(arr)();
}

// Alternative if adapt doesn't work
double sum_list_alternative(py::list py_list) {
    std::vector<double> vec;
    for (auto item : py_list) {
        vec.push_back(item.cast<double>());
    }

    // Create xarray directly
    xt::xarray<double> arr = xt::zeros<double>({vec.size()});
    for (size_t i = 0; i < vec.size(); ++i) {
        arr(i) = vec[i];
    }
    return xt::sum(arr)();
}

PYBIND11_MODULE(_fast_module, m) {
    m.doc() = "Fast C++ module built with MinGW";

    m.def("add_arrays", &add_arrays, "Add two xtensor arrays");
    m.def("sum_list", &sum_list, "Sum a Python list");

    // Simple test function
    m.def("hello", []() { return "Hello from MinGW-compiled C++ extension!"; });
}