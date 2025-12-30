cmake_minimum_required(VERSION 3.10)
project(_cpp_force_kernel LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- Python setup ---
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Dynamically get NumPy include path
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "Using Python include: ${Python_INCLUDE_DIRS}")
message(STATUS "Using Python library: ${Python_LIBRARIES}")
message(STATUS "Using NumPy include: ${NUMPY_INCLUDE}")

# --- Create module ---
add_library(_cpp_force_kernel MODULE main.cpp)

# --- Performance flags ---
if(MSVC)
    target_compile_options(_cpp_force_kernel PRIVATE
        /O2
        /fp:fast
    )
else()
    target_compile_options(_cpp_force_kernel PRIVATE
        -O3
        -ffast-math
        -march=native
    )
endif()

# --- Include directories ---
target_include_directories(_cpp_force_kernel PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/xtensor/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/xtensor-python/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/xtl/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/pybind11/include
    ${Python_INCLUDE_DIRS}
    ${NUMPY_INCLUDE}
)

# --- Compile definitions for header-only xtensor-python ---
# target_compile_definitions(_cpp_force_kernel PRIVATE XTENSOR_PYTHON_HEADER_ONLY)

# --- Link Python library ---
target_link_libraries(_cpp_force_kernel PRIVATE ${Python_LIBRARIES})

# --- MSVC-specific options ---
if(MSVC)
    target_compile_options(_cpp_force_kernel PRIVATE /permissive- /Zc:__cplusplus)
endif()

# --- Output settings ---
set_target_properties(_cpp_force_kernel PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_cpp_force_kernel"
    SUFFIX ".pyd"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../project/simulation/cpp_force_kernel
)
