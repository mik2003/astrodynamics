cmake_minimum_required(VERSION 3.10)
project(_fast_module)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(PYTHON_INCLUDE_DIR "C:/Users/User/AppData/Local/Python/pythoncore-3.13-64/include")
set(PYTHON_LIBRARY "C:/Users/User/AppData/Local/Python/pythoncore-3.13-64/libs/python313.lib")

add_library(_fast_module MODULE main.cpp)

target_include_directories(_fast_module PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/xtensor/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/xtl/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/pybind11/include
    ${PYTHON_INCLUDE_DIR}
)

target_link_libraries(_fast_module PRIVATE ${PYTHON_LIBRARY})

set_target_properties(_fast_module PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_fast_module"
    SUFFIX ".pyd"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../project/fast_module
)
