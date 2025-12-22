"""Integrator module"""

from typing import Callable, Concatenate

import numpy as np

from project.utils import A, P


class Integrator:
    @staticmethod
    def euler(
        state: A,
        time_step: float,
        stop_time: float,
        func: Callable[Concatenate[A, P], A],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> A:
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
        # Initialize full state vector
        y = np.zeros((steps, state.size))
        # Change print_step for debugging
        # progress = ProgressTracker(n=steps, print_step=10000, name="Integrating Euler")
        # Perform Euler integration
        y[0, :] = state
        for i in range(steps - 1):
            y[i + 1, :] = y[i, :] + time_step * func(y[i, :], *args, **kwargs)
            # Uncomment for debugging
            # progress.print(i=i)
        # progress.print(i=steps)

        return y

    @staticmethod
    def rk4(
        state: A,
        time_step: float,
        stop_time: float,
        func: Callable[Concatenate[A, P], A],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> A:
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
        # Initialize full state vector
        y = np.zeros((steps, state.size))
        # Change print_step for debugging
        # progress = ProgressTracker(n=steps, print_step=10000, name="Integrating RK4")
        # Perform RK4 integration
        y[0, :] = state
        for i in range(steps - 1):
            k_1 = func(y[i, :], *args, **kwargs)
            k_2 = func(y[i, :] + k_1 * time_step / 2, *args, **kwargs)
            k_3 = func(y[i, :] + k_2 * time_step / 2, *args, **kwargs)
            k_4 = func(y[i, :] + k_3 * time_step, *args, **kwargs)
            y[i + 1, :] = y[i, :] + time_step / 6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
            # Uncomment for debugging
            # progress.print(i=i)
        # progress.print(i=steps)

        return y
