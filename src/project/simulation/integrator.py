"""Integrator module"""

from typing import Callable, Concatenate, Protocol, runtime_checkable

import numpy as np

from project.utils import FloatArray, P, ProgressTracker


@runtime_checkable
class FunctionProtocol(Protocol[P]):
    def __call__(
        self, state: FloatArray, out: FloatArray, *args: P.args, **kwargs: P.kwargs
    ) -> None: ...


@runtime_checkable
class RK4Capable(Protocol[P]):
    def _rk4_backend(
        self,
        state: FloatArray,
        time_step: float,
        stop_time: float,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> FloatArray: ...


@runtime_checkable
class EulerCapable(Protocol[P]):
    def _euler_backend(
        self,
        state: FloatArray,
        time_step: float,
        stop_time: float,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> FloatArray: ...


class Integrator:
    @staticmethod
    def euler(
        state: FloatArray,
        time_step: float,
        stop_time: float,
        func: FunctionProtocol[P],
        progress: bool = True,
        print_step: int = 10_000,
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
        func : FunctionProtocol[P]
            Function to integrate
        progress : bool, optional
            Wether to print progress, by default True
        print_step : int
            Interval between progress reports, by default 10_000

        Returns
        -------
        A
            Integrated function
        """
        # Check for function specific implementation
        if isinstance(func, EulerCapable):
            return func._euler_backend(state, time_step, stop_time, *args, **kwargs)

        # Default to numpy implementation
        return Integrator._euler(
            state, time_step, stop_time, func, progress, print_step, *args, **kwargs
        )

    @staticmethod
    def rk4(
        state: FloatArray,
        time_step: float,
        stop_time: float,
        func: FunctionProtocol[P],
        progress: bool = True,
        print_step: int = 10_000,
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
        func : FunctionProtocol[P]
            Function to integrate
        progress : bool, optional
            Wether to print progress, by default True
        print_step : int
            Interval between progress reports, by default 10_000

        Returns
        -------
        A
            Integrated function
        """
        # Check for function specific implementation
        if isinstance(func, RK4Capable):
            return func._rk4_backend(state, time_step, stop_time, *args, **kwargs)

        # Default to numpy implementation
        return Integrator._rk4(
            state, time_step, stop_time, func, progress, print_step, *args, **kwargs
        )

    @staticmethod
    def _euler(
        state: FloatArray,
        time_step: float,
        stop_time: float,
        func: Callable[Concatenate[FloatArray, FloatArray, P], None],
        progress: bool = True,
        print_step: int = 10_000,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> FloatArray:
        """Euler integrator (numpy implementation)

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
        progress : bool, optional
            Wether to print progress, by default True
        print_step : int
            Interval between progress reports, by default 10_000

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
        if progress:
            pt = ProgressTracker(
                n=steps, print_step=print_step, name="Integrating Euler"
            )
        # Perform Euler integration
        y[0, :] = state

        # Euler buffer
        tmp = np.empty(dim)

        for i in range(steps - 1):
            func(y[i, :], tmp, *args, **kwargs)
            y[i + 1, :] = y[i, :] + time_step * tmp
            if progress:
                pt.print(i=i)
        if progress:
            pt.print(i=steps)

        return y

    @staticmethod
    def _rk4(
        state: FloatArray,
        time_step: float,
        stop_time: float,
        func: Callable[Concatenate[FloatArray, FloatArray, P], None],
        progress: bool = True,
        print_step: int = 10_000,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> FloatArray:
        """Runge-Kutta 4 integrator (numpy implementation)

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
        progress : bool, optional
            Wether to print progress, by default True
        print_step : int
            Interval between progress reports, by default 10_000

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
        if progress:
            pt = ProgressTracker(n=steps, print_step=print_step, name="Integrating RK4")
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
            if progress:
                pt.print(i=i)
        if progress:
            pt.print(i=steps)

        return y
