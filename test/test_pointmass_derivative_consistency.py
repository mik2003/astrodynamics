import numpy as np
import pytest

from project.simulation.model import (
    CPPPointMass,
    NumbaPointMass,
    NumpyPointMass,
)
from project.utils import Dir
from project.utils.data import BodyList


@pytest.fixture(scope="module")
def solar_system_state():
    """
    Load a realistic N-body test state once per module.
    """
    bl = BodyList.load(Dir.data / "solar_system_2460967.toml")
    return bl.y_0.copy(), bl.n, bl.mu


@pytest.fixture(scope="module")
def output_buffers(solar_system_state):
    """
    Preallocate output buffers for all kernels.
    """
    state, _, _ = solar_system_state
    return (
        np.empty_like(state),
        np.empty_like(state),
        np.empty_like(state),
    )


def test_pointmass_single_derivative_consistency(
    solar_system_state,
    output_buffers,
):
    """
    All point-mass kernels must produce the same time derivative
    for the same state vector.
    """
    state, n, mu = solar_system_state
    out_numpy, out_numba, out_cpp = output_buffers

    # Evaluate derivatives (NO integrator)
    NumpyPointMass()(state, out=out_numpy, n=n, mu=mu)
    NumbaPointMass()(state, out=out_numba, n=n, mu=mu)
    CPPPointMass()(state, out=out_cpp, n=n, mu=mu)

    rtol = 1e-13
    atol = 1e-13

    # NumPy vs Numba
    np.testing.assert_allclose(
        out_numpy,
        out_numba,
        rtol=rtol,
        atol=atol,
        err_msg="NumPy and Numba kernels disagree on single derivative",
    )

    # NumPy vs C++
    np.testing.assert_allclose(
        out_numpy,
        out_cpp,
        rtol=rtol,
        atol=atol,
        err_msg="NumPy and C++ kernels disagree on single derivative",
    )
