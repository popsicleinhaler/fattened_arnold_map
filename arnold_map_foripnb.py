"""
Project file for QLSC 600 Module 1.

Chosen Project: 5B
"""

from numpy import sin, pi, ceil, inf, matrix, zeros, ndarray

import numpy as np
import math

from skimage.io import imshow, imsave

from matplotlib import pyplot as plt
from typing import List, Optional, Set

def plot_2d_phase_portrait(a: float=1, k: float=0.5, T_osc: float=24, tau_0: float=1,
                           t_0_min: float = -1, t_0_max: float = 10, t_0_step: float = 0.5,
                           T_0_min: float = -1, T_0_max: float = 10, T_0_step: float = 0.5, n_f: int = 1000):
    
    #-----------------------------------------------------------------
    def perform_simulation(T_0: float, t_0: float, n_f: int, a: float, k: float, T_osc: float, tau_0: float) \
        -> (List[float], List[float]):
    
    #-----------------------
        def one_iterate(t_n: float, T_n: float, tau_0: float, a: float, k: float, T_osc: float) -> (float, float):
            """
            Perform a single iterate of the map.

            :param t_n:
            :param T_n:
            :param tau_0:
            :param a:
            :param k:
            :param T_osc:
            :return: Two floats, corresponding to T_{n+1} and t_{n+1}
            """
            t = (T_n + t_n) % T_osc
            T = tau_0 * (1 - a) + a * T_n + k * sin(2 * pi * t / T_osc)

            return T, t
        #-----------------------


        t = t_0
        T = T_0
        t_values = [t_0]
        T_values = [T_0]

        # while t < t_f:
        for i in range(1, n_f):
            T, t = one_iterate(t, T, tau_0, a, k, T_osc)

            t_values.append(t)
            T_values.append(T)

        return T_values, t_values
    #-----------------------------------------------------------------

    t_0_values: List[float] = []
    T_0_values: List[float] = []

    t_0 = t_0_min
    T_0 = T_0_min

    while t_0 < t_0_max:
        t_0_values.append(t_0)
        t_0 += t_0_step

    while T_0 < T_0_max:
        T_0_values.append(T_0)
        T_0 += T_0_step

    for t_0 in t_0_values:
        for T_0 in T_0_values:
            T_values, t_values = perform_simulation(T_0, t_0, n_f, a, k, T_osc, tau_0)
            plt.plot(t_values, T_values, '.', markersize=1)

    plt.show()
    


def plot_simulation(t_0: float, n_f: int, T_0: float, a: float, k: float, T_osc: float, tau_0: float, epsilon: float = 1e-4):
    T_values, t_values = perform_simulation(T_0, t_0, n_f, a, k, T_osc, tau_0)

    periodicity = determine_periodicity(T_values, t_values, epsilon=epsilon)
    print("We have periodicity {}".format(periodicity))

    n_values: List[int] = list(range(0, n_f))

    plt.plot(n_values, T_values, 'b*-', label=r"$T_n$")
    plt.xlabel(r"$n$")
    plt.ylabel(r"$T_n$")
    plt.title(r"Time Series for $T_n$")
    number_of_stimuli = int(ceil(n_f / T_osc))
    first_stimulus = 0  # t_0 // T_osc * T_osc
    stimuli = [first_stimulus + i * T_osc for i in range(0, number_of_stimuli)]
    min_y = min(T_values)
    max_y = max(T_values)
    plt.vlines(stimuli, min_y, max_y, linestyles='dotted', colors='red')
    plt.show()
    plt.plot(n_values, t_values, 'b*-', label=r"$t_n$")
    plt.ylabel(r"$t_n$")
    plt.xlabel(r"$n$")
    plt.title(r"Time Series for $t_n$")
    min_y = min(t_values)
    max_y = max(t_values)
    plt.vlines(stimuli, min_y, max_y, linestyles='dotted', colors='red')
    plt.show()