"""Integrator module"""

from typing import Callable, Concatenate

import numpy as np

from project.utils import FloatArray, P, ProgressTracker


class Integrator:
    @staticmethod
    def euler(
        state: FloatArray,
        time_step: float,
        stop_time: float,
        func: Callable[Concatenate[FloatArray, FloatArray, P], None],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> FloatArray:
        """Euler integrator

        Parameters
        ----------
        state : A
            Initial state vector
        time_step : float
            Time step [s]
        stop_time : float
            Stop time [s]
        func : Callable[Concatenate[A, P], A]
            Function to integrate

        Returns
        -------
        A
            Integrated function
        """
        # Define time array
        time_array_seconds = np.arange(
            start=0,
            stop=stop_time + time_step,  # Include last point
            step=time_step,
            dtype=np.float64,
        )
        steps = time_array_seconds.size
        dim = state.size
        # Initialize full state vector
        y = np.zeros((steps, dim))
        # Change print_step for debugging
        progress = ProgressTracker(n=steps, print_step=10000, name="Integrating Euler")
        # Perform Euler integration
        y[0, :] = state

        # Euler buffer
        tmp = np.empty(dim)

        for i in range(steps - 1):
            func(y[i, :], tmp, *args, **kwargs)
            y[i + 1, :] = y[i, :] + time_step * tmp
            # Uncomment for debugging
            progress.print(i=i)
        progress.print(i=steps)

        return y

    @staticmethod
    def rk4(
        state: FloatArray,
        time_step: float,
        stop_time: float,
        func: Callable[Concatenate[FloatArray, FloatArray, P], None],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> FloatArray:
        """Runge-Kutta 4 integrator

        Parameters
        ----------
        state : A
            Initial state vector
        time_step : float
            Time step [s]
        stop_time : float
            Stop time [s]
        func : Callable[Concatenate[A, P], A]
            Function to integrate

        Returns
        -------
        A
            Integrated function
        """
        if hasattr(func, "_rk4_backend"):
            return func._rk4_backend(state, time_step, stop_time, *args, **kwargs)

        return _rk4_numpy(state, time_step, stop_time, func, *args, **kwargs)


def _rk4_numpy(
    state: FloatArray,
    time_step: float,
    stop_time: float,
    func: Callable[Concatenate[FloatArray, FloatArray, P], None],
    *args: P.args,
    **kwargs: P.kwargs,
) -> FloatArray:
    """Runge-Kutta 4 integrator

    Parameters
    ----------
    state : A
        Initial state vector
    time_step : float
        Time step [s]
    stop_time : float
        Stop time [s]
    func : Callable[Concatenate[A, P], A]
        Function to integrate

    Returns
    -------
    A
        Integrated function
    """
    # Define time array
    time_array_seconds = np.arange(
        start=0,
        stop=stop_time + time_step,  # Include last point
        step=time_step,
        dtype=np.float64,
    )
    steps = time_array_seconds.size
    dim = state.size
    # Initialize full state vector
    y = np.zeros((steps, dim))
    # Change print_step for debugging
    progress = ProgressTracker(n=steps, print_step=10000, name="Integrating RK4")
    # Perform RK4 integration
    y[0, :] = state

    # RK buffers
    k1 = np.empty(dim)
    k2 = np.empty(dim)
    k3 = np.empty(dim)
    k4 = np.empty(dim)

    for i in range(steps - 1):
        func(y[i, :], k1, *args, **kwargs)
        func(y[i, :] + k1 * time_step / 2, k2, *args, **kwargs)
        func(y[i, :] + k2 * time_step / 2, k3, *args, **kwargs)
        func(y[i, :] + k3 * time_step, k4, *args, **kwargs)
        y[i + 1, :] = y[i, :] + time_step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        # Uncomment for debugging
        progress.print(i=i)
    progress.print(i=steps)

    return y
