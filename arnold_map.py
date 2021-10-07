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

from threading import Thread
from multiprocessing import Process


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


def perform_simulation(T_0: float, t_0: float, n_f: int, a: float, k: float, T_osc: float, tau_0: float) \
        -> (List[float], List[float]):
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


def approximate_equality(a: float, b: float, epsilon: float = 1e-4) -> bool:
    # print("Comparing {} and {} with epsilon={}".format(a, b, epsilon))
    return abs(a - b) < epsilon


def plot_2d_phase(t_0: float, n_f: int, T_0: float, a: float, k: float, T_osc: float, tau_0: float):
    T_values, t_values = perform_simulation(T_0, t_0, n_f, a, k, T_osc, tau_0)

    plt.xlabel(r"t_n")
    plt.ylabel(r"T_n")
    plt.title(r"Phase Diagram $T_n$ vs. $t_n$")

    plt.plot(t_values, T_values, "b*")
    plt.show()


def plot_2d_phase_portrait(a: float, k: float, T_osc: float, tau_0: float,
                           t_0_min: float = -1, t_0_max: float = 1, t_0_step: float = 0.1,
                           T_0_min: float = -1, T_0_max: float = 1, T_0_step: float = 0.1, n_f: int = 1000):

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
            plt.plot(t_values, T_values, '*')

    plt.show()

# def determine_periodicity(T_values: List[float], max_period: int = 100, start_index: int = 5) -> Optional[int]:
#     # for i in range(1, max_period):
#     #     if start_index + i >= len(T_values):
#     #         return None
#     #
#     #     if approximate_equality(T_values[start_index], T_values[start_index + i], epsillon=0.1):
#     #         return i
#     #
#     # return inf
#     # values: Set[float] = set()
#
#     mutable_T_values = list(T_values[start_index:])
#
#     x = mutable_T_values[0]
#
#     j = 0
#
#     for i, v in enumerate(mutable_T_values):
#         if i == 0:
#             continue
#
#         if approximate_equality(v, x, epsillon=0.01):
#             j = i
#             print("Found j={}".format(j))
#             break
#
#     if j == 0:
#         return None
#
#     mutable_T_values = mutable_T_values[0:j]
#
#     extrema: Set[float] = set()
#
#     for i in range(1, len(mutable_T_values) - 1):
#         a = mutable_T_values[i-1]
#         b = mutable_T_values[i]
#         c = mutable_T_values[i+1]
#
#         if (b > a and b > c) or (b < a and b < c):
#             extrema.add(b)
#
#     return len(extrema)
#
#     # while len(mutable_T_values) > 0:
#     #     top = mutable_T_values.pop(0)
#     #
#     #     values.add(top)
#     #
#     #     for value in mutable_T_values:
#     #         if approximate_equality(top, value, epsillon=0.01):
#     #             mutable_T_values.remove(value)
#     #
#     # return len(values)


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


def plot_a_bifurcations(T_0: float, t_0: float, n_f: int, k: float, T_osc: float, tau_0: float,
                        min_a: float = -1, max_a: float = 1, step_a: float = 0.1, num_iterates_for_diagram: Optional[int] = None, epsilon: float = 1e-2):
    a_values: List[float] = []
    t_results: List[List[float]] = []
    T_results: List[List[float]] = []

    a = min_a

    while a < max_a:
        a_values.append(a)
        a = a + step_a

    use_period: bool = num_iterates_for_diagram is None

    periodicities: List[Optional[int]] = [] if use_period else [num_iterates_for_diagram for _ in a_values]

    for a in a_values:
        T_values, t_values = perform_simulation(T_0, t_0, n_f, a, k, T_osc, tau_0)
        t_results.append(t_values)
        T_results.append(T_values)

        if use_period:
            periodicity = determine_periodicity(T_values, t_values, epsilon=epsilon)
            periodicities.append(periodicity)

    print(a_values)

    # final_T_values = [T_values[-1] for T_values in T_results]
    # late_T_values: List[List[float]] = [iterate_result[-num_iterates_for_diagram:-1] for iterate_result in T_results]

    # print(late_T_values)

    # a_values_multiplied: List[float] = []
    # final_T_values: List[float] = []

    # for i, sublist in enumerate(late_T_values):
    #     a_values_multiplied.extend(len(sublist) * [a_values[i]])
    #     final_T_values.extend(sublist)

    for i in range(0, len(T_results)):
        a = a_values[i]
        T_values = T_results[i]
        period = periodicities[i]

        if period is None:
            period = len(T_values) - 1

        T_values_to_plot: List[float] = T_values[-period - 1:-1]
        a_values_to_plot: List[float] = period * [a]

        plt.plot(a_values_to_plot, T_values_to_plot, '*')


    # plt.plot(a_values_multiplied, final_T_values, 'b*')
    plt.title(r"$T_n$ vs. $\alpha$ Bifurcations")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$T_n$")
    plt.show()

    for i in range(0, len(t_results)):
        a = a_values[i]
        t_values = t_results[i]
        period = periodicities[i]

        if period is None:
            period = len(t_values) - 1

        t_values_to_plot: List[float] = t_values[-period - 1:-1]
        a_values_to_plot: List[float] = period * [a]

        plt.plot(a_values_to_plot, t_values_to_plot, '*')

    plt.title(r"$t_n$ vs. $\alpha$ Bifurcations")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$t_n$")
    plt.show()


def plot_tau_0_bifurcations(T_0: float, t_0: float, n_f: int, a: float, k: float, T_osc: float,
                            min_tau_0: float = 0, max_tau_0: float = 30, step_tau_0: float = 0.5,
                            num_iterates_for_diagram: Optional[int] = None, epsilon: float = 1e-2):
    tau_0_values: List[float] = []
    t_results: List[List[float]] = []
    T_results: List[List[float]] = []
    # periodicities: List[Optional[int]] = []

    tau_0 = min_tau_0

    while tau_0 < max_tau_0:
        tau_0_values.append(tau_0)
        tau_0 = tau_0 + step_tau_0

    use_period: bool = num_iterates_for_diagram is None

    periodicities: List[Optional[int]] = [] if use_period else [num_iterates_for_diagram for _ in tau_0_values]

    for tau_0 in tau_0_values:
        T_values, t_values = perform_simulation(T_0, t_0, n_f, a, k, T_osc, tau_0)
        t_results.append(t_values)
        T_results.append(T_values)

        if use_period:
            period = determine_periodicity(T_values, t_values, epsilon)
            periodicities.append(period)

    print(tau_0_values)

    # final_T_values = [T_values[-1] for T_values in T_results]

    for i in range(0, len(T_results)):
        tau_0 = tau_0_values[i]
        T_values = T_results[i]
        period = periodicities[i]

        if period is None:
            period = len(T_values) - 1

        T_values_to_plot: List[float] = T_values[-period - 1:-1]
        tau_0_values_to_plot: List[float] = period * [tau_0]

        plt.plot(tau_0_values_to_plot, T_values_to_plot, '*')


    # plt.plot(a_values_multiplied, final_T_values, 'b*')
    plt.title(r"$T_n$ vs. $\tau_0$ Bifurcations")
    plt.xlabel(r"$\tau_0$")
    plt.ylabel(r"$T_n$")
    plt.show()

    for i in range(0, len(t_results)):
        tau_0 = tau_0_values[i]
        t_values = t_results[i]
        period = periodicities[i]

        if period is None:
            period = len(t_values) - 1

        t_values_to_plot: List[float] = t_values[-period-1:-1]
        tau_0_values_to_plot: List[float] = period * [tau_0]

        plt.plot(tau_0_values_to_plot, t_values_to_plot, '*')

    plt.title(r"$t_n$ vs. $\tau_0$ Bifurcations")
    plt.xlabel(r"$\tau_0$")
    plt.ylabel(r"$t_n$")
    plt.show()

    # late_T_values: List[List[float]] = [iterate_result[-num_iterates_for_diagram:-1] for iterate_result in T_results]
    # late_t_values: List[List[float]] = [iterate_result[-num_iterates_for_diagram:-1] for iterate_result in t_results]
    #
    # # print(late_T_values)
    #
    # tau_0_values_multiplied: List[float] = []
    # final_T_values: List[float] = []
    # final_t_values: List[float] = []
    #
    # for i, (sublist_T, sublist_t) in enumerate(zip(late_T_values, late_t_values)):
    #     tau_0_values_multiplied.extend(len(sublist_T) * [tau_0_values[i]])
    #     final_T_values.extend(sublist_T)
    #     final_t_values.extend(sublist_t)
    #
    # plt.plot(tau_0_values_multiplied, final_T_values, 'b*')
    # plt.show()
    #
    # plt.plot(tau_0_values_multiplied, final_t_values, 'b*')
    # plt.show()


def plot_k_bifurcations(T_0: float, t_0: float, n_f: int, a: float, T_osc: float, tau_0: float,
                        min_k: float = 0, max_k: float = 1, step_k: float = 0.1, num_iterates_for_diagram : Optional[int] = None, epsilon: float = 1e-2):
    k_values: List[float] = []
    t_results: List[List[float]] = []
    T_results: List[List[float]] = []
    # periodicities: List[Optional[int]] = []

    k = min_k

    while k < max_k:
        k_values.append(k)
        k = k + step_k

    use_period: bool = num_iterates_for_diagram is None

    periodicities: List[Optional[int]] = [] if use_period else [num_iterates_for_diagram for _ in k_values]

    for k in k_values:
        T_values, t_values = perform_simulation(T_0, t_0, n_f, a, k, T_osc, tau_0)
        t_results.append(t_values)
        T_results.append(T_values)

        if use_period:
            period = determine_periodicity(T_values,t_values, epsilon)
            periodicities.append(period)

    print(k_values)

    # final_T_values = [T_values[-1] for T_values in T_results]

    # late_T_values: List[List[float]] = [iterate_result[-num_iterates_for_diagram:-1] for iterate_result in T_results]

    # print(late_T_values)

    for i in range(0, len(T_results)):
        k = k_values[i]
        T_values = T_results[i]
        period = periodicities[i]

        if period is None:
            # continue
            period = len(T_values) - 1

        T_values_to_plot: List[float] = T_values[-period - 1:-1]
        k_values_to_plot: List[float] = period * [k]

        plt.plot(k_values_to_plot, T_values_to_plot, '*')


    # plt.plot(a_values_multiplied, final_T_values, 'b*')
    plt.title(r"$T_n$ vs. $k$ Bifurcations")
    plt.xlabel(r"$k$")
    plt.ylabel(r"$T_n$")
    plt.show()

    for i in range(0, len(t_results)):
        k = k_values[i]
        t_values = t_results[i]
        period = periodicities[i]

        if period is None:
            period = len(t_values) - 1

        t_values_to_plot: List[float] = t_values[-period-1:-1]
        k_values_to_plot: List[float] = period * [k]

        plt.plot(k_values_to_plot, t_values_to_plot, '*')

    plt.title(r"$t_n$ vs. $k$ Bifurcations")
    plt.xlabel(r"$k$")
    plt.ylabel(r"$t_n$")
    plt.show()

    # k_values_multiplied: List[float] = []
    # final_T_values: List[float] = []
    # 
    # for i, sublist in enumerate(late_T_values):
    #     k_values_multiplied.extend(len(sublist) * [k_values[i]])
    #     final_T_values.extend(sublist)
    # 
    # plt.plot(k_values_multiplied, final_T_values, 'b*')
    # plt.show()


def logistic_map(x: float, r: float) -> float:
    return r * x * (1 - x)


def logistic_map_simulation(x_0: float, r: float, num_steps: int = 100) -> List[float]:
    x_values = [x_0]

    x = x_0

    for i in range(1, num_steps):
        x = logistic_map(x, r)
        x_values.append(x)

    return x_values


def logistic_map_bifurcations(x_0: float, r_min: float = 0, r_max: float = 3.999,
                              r_step: float = 0.1, num_iterations: int = 100):
    r_values: List[float] = []
    iterate_results: List[List[float]] = []

    r: float = r_min

    while r < r_max:
        r_values.append(r)
        r += r_step

    for r in r_values:
        iterate_results.append(logistic_map_simulation(x_0, r, num_steps=num_iterations))

    late_values: List[List[float]] = [iterate_result[-20:-1] for iterate_result in iterate_results]

    print(late_values)

    r_values_multiplied: List[float] = []
    final_values: List[float] = []

    for i, sublist in enumerate(late_values):
        r_values_multiplied.extend(len(sublist) * [r_values[i]])
        final_values.extend(sublist)

    plt.plot(r_values_multiplied, final_values, 'b*')
    plt.show()


def determine_periodicity(T_values: List[float], t_values: List[float], epsilon: float = 5e-2,
                          use_max: bool = True) -> Optional[int]:

    transients = round(len(T_values) / 8)

    # print("Getting rid of {} transients".format(transients))

    last_T_values = T_values[transients:]
    last_t_values = t_values[transients:]

    zipped_values = list(zip(last_T_values, last_t_values))

    max_T = max(last_T_values[:-1]) if use_max else min(last_T_values[:-1])
    # min_T = max(last_T_values) if not use_max else min(last_T_values)

    # epsilon = abs((max_T - min_T)/10) + 1e-2

    # epsilon = 5e-2

    # print("Using epsilon={}".format(epsilon))

    index_of_max = last_T_values[:-1].index(max_T)

    # print("Index of max is {}".format(index_of_max))

    max_t = last_t_values[index_of_max]

    # print("Starting at index {}".format(index_of_max))

    # print("Values under consideration are T={} and t={}".format(max_T, max_t))

    increment = 1
    end = len(zipped_values) - 1

    if end - index_of_max < 5:
        increment = -1
        end = 0

    start = index_of_max + increment

    # periodicity = None

    # last_T, last_t = max(last_T_values) #zipped_values[-1]

    # one_before_max_T = last_T_values[index_of_max - 1]
    # one_before_max_t = last_t_values[index_of_max - 1]
    # one_after_max_T = last_T_values[index_of_max + 1]
    # one_after_max_t = last_t_values[index_of_max + 1]

    periodicity = None

    # print("Values to match: T - {}, t - {}".format(max_T, max_t))

    for i in range(start, end, increment):
        # print("Considering i={}".format(i))
        new_T, new_t = zipped_values[i]

        # print("Considering index {}".format(i))

        # print("Differences: T - {}, t - {}".format(abs(new_T-max_T), abs(new_t - max_t)))

        if abs(new_T - max_T) < epsilon and abs(new_t - max_t) < epsilon:

            test_periodicity = abs(index_of_max - i)

            # print("Potential match: period {}".format(test_periodicity))

            for j in range(0, len(zipped_values) - test_periodicity):
                current_T, current_t = zipped_values[j]
                future_T, future_t = zipped_values[j + test_periodicity]

                if not (approximate_equality(current_T, future_T, epsilon) and approximate_equality(current_t, future_t,
                                                                                                    epsilon)):
                    periodicity = None
                    break

                periodicity = test_periodicity
                break

            if periodicity is not None:
                break

            # prev_T, prev_t = zipped_values[i-1]
            # next_T, next_t = zipped_values[i+1]
            #
            # if approximate_equality(prev_T, one_before_max_T, epsilon) \
            #         and approximate_equality(prev_t, one_before_max_t, epsilon)\
            #         and approximate_equality(next_T, one_after_max_T, epsilon)\
            #         and approximate_equality(next_t, one_after_max_t, epsilon):
            #     periodicity = abs(index_of_max - i)
            #     break

            # return -(i + 1)

    if periodicity is not None:
        return periodicity

    increment = -increment
    end = len(zipped_values) - 1 if increment == 1 else 0

    start = index_of_max + increment

    for i in range(start, end, increment):
        # print("Considering index {}".format(i))
        new_T, new_t = zipped_values[i]

        if abs(new_T - max_T) < epsilon and abs(new_t - max_t) < epsilon:

            test_periodicity = abs(index_of_max - i)

            for j in range(0, len(zipped_values) - test_periodicity):
                current_T, current_t = zipped_values[j]
                future_T, future_t = zipped_values[j + test_periodicity]

                if not (approximate_equality(current_T, future_T, epsilon) and approximate_equality(current_t, future_t, epsilon)):
                    periodicity = None
                    break

                periodicity = test_periodicity


            # prev_T, prev_t = zipped_values[i-1]
            # next_T, next_t = zipped_values[i+1]

            # if approximate_equality(prev_T, one_before_max_T, epsilon) \
            #         and approximate_equality(prev_t, one_before_max_t, epsilon)\
            #         and approximate_equality(next_T, one_after_max_T, epsilon)\
            #         and approximate_equality(next_t, one_after_max_t, epsilon):
            # periodicity = test_periodicity#abs(index_of_max - i)

    return periodicity

    # final_values = T_values[transients:]
    # max_value = max(final_values) if use_max else min(final_values)
    #
    # index_of_max = final_values.index(max_value)
    #
    # increment = 1
    # end = len(final_values)
    #
    # if end - index_of_max < 5:
    #     increment = -1
    #     end = -1
    #
    # start = index_of_max + increment

    # periodicity = None
    #
    # for i in range(start, end, increment):
    #     if approximate_equality(max_value, final_values[i], epsilon=epsilon):
    #         periodicity = abs(index_of_max - i)
    #         break

    # if periodicity is not None:
    #     if not approximate_equality(max_value, final_values[index_of_max + 2*increment], epsilon=epsilon):
    #         # periodicity = determine_periodicity(T_values, epsilon, transients, not use_max)
    #         pass

    # return periodicity


def two_dimensional_bifurcation(filename: str, T_0: float, t_0: float, n_f: int, a: float, T_osc: float,
                                min_k: float = 0, max_k: float = 1, step_k: float = 0.005,
                                min_tau_0: float = 0, max_tau_0: float = 30, step_tau_0: float = 0.005,
                                epsilon: float = 1e-2):
    # T_results: List[List[float]] = []
    # t_results: List[List[float]] = []
    k_values: List[float] = []
    tau_0_values: List[float] = []

    k = min_k

    while k < max_k:
        k_values.append(k)
        k = k + step_k

    tau_0 = min_tau_0

    while tau_0 < max_tau_0:
        tau_0_values.append(tau_0)
        tau_0 = tau_0 + step_tau_0

    periodicity_grid: np.ndarray = np.zeros(shape=(len(k_values), len(tau_0_values), 1), dtype=np.uint16)

    white = 2**16 - 1
    gray_factor = math.floor(white / (n_f - 1))

    for i, k in enumerate(k_values):

        if not i % 10:
            print("Row progress: {}%".format(round(i / len(k_values) * 100, 2)))

        for j, tau_0 in enumerate(tau_0_values):
            if not j % 20:
                print("Column progress: {}%".format(round(j / len(tau_0_values) * 100, 2)))

            T_values, t_values = perform_simulation(T_0, t_0, n_f, a, k, T_osc, tau_0)

            periodicity = determine_periodicity(T_values, t_values, epsilon=epsilon)

            print("Found period {}".format(periodicity))

            if periodicity is None:
                periodicity = 0
            else:
                # periodicity = 255 - 2 * (periodicity - 1)
                periodicity = white - gray_factor * (periodicity - 1)

                if periodicity < 0:
                    print("ERROR!!!! NEGATIVE PERIODICITY {}!".format(periodicity))
                    periodicity = 0

            periodicity_grid[i, j] = periodicity

        # plt.imshow(periodicity_grid)
        # plt.show(block=False)

    # print("We have periodicity grid:")
    # print(periodicity_grid.tostring())

    # imshow(periodicity_grid, plugin='matplotlib')
    plt.imshow(periodicity_grid)
    imsave(filename, arr=periodicity_grid)


def __run_iterates_for_a_values(a_values_list: List[int], starting_index: int, base_filename: str, T_0: float,
                                t_0: float, n_f: int, T_osc: float,
                                min_tau_0: float, max_tau_0: float, step_tau_0: float,
                                min_k: float, max_k: float, step_k: float, epsilon: float):
    for i, a in enumerate(a_values_list):
        filename_a_string = "{:03d}_a_{}".format(i + starting_index, round(a, 6))
        fn = base_filename.format(filename_a_string)
        two_dimensional_bifurcation(fn, T_0=T_0, t_0=t_0, n_f=n_f, a=a, T_osc=T_osc,
                                    min_tau_0=min_tau_0, max_tau_0=max_tau_0, step_tau_0=step_tau_0,
                                    min_k=min_k, max_k=max_k, step_k=step_k, epsilon=epsilon)
        print("Written {}".format(fn))


def full_two_dimensional_bifurcation_stack(base_filename: str, T_0: float, t_0: float, n_f: int, T_osc: float,
                                           min_k: float = 0, max_k: float = 1, step_k: float = 0.005,
                                           min_tau_0: float = 0, max_tau_0: float = 30, step_tau_0: float = 0.005,
                                           epsilon: float = 1e-2, min_a: float = -1, max_a: float = 1,
                                           step_a: float = 0.01, reverse_a: bool = False, use_processes: bool = True,
                                           num_cores: int = 4):
    a_values: List[float] = []

    a = min_a

    while a < max_a:
        a_values.append(a)
        a += step_a

    if reverse_a:
        a_values.reverse()

    if use_processes:

        frames_per_core = int(ceil(len(a_values)/num_cores))

        for j in range(0, len(a_values), frames_per_core):
            a_sublist = a_values[j:j+frames_per_core]

            Process(target=__run_iterates_for_a_values, kwargs={
                "a_values_list": a_sublist,
                "starting_index": j,
                "base_filename": base_filename,
                "T_0": T_0,
                "t_0": t_0,
                "n_f": n_f,
                "T_osc": T_osc,
                "min_tau_0": min_tau_0,
                "max_tau_0": max_tau_0,
                "step_tau_0": step_tau_0,
                "min_k": min_k,
                "max_k": max_k,
                "step_k": step_k,
                "epsilon": epsilon
            }).start()

    else:

        for i, a in enumerate(a_values):
            if not i % 5:
                print("Layer progress: {}%".format(round(i/len(a_values)*100, 2)))

            filename_a_string = "a_{}".format(round(a, 2))
            fn = base_filename.format(filename_a_string)
            two_dimensional_bifurcation(fn, T_0=T_0, t_0=t_0, n_f=n_f, a=a, T_osc=T_osc, min_k=min_k, max_k=max_k, step_k=step_k, min_tau_0=min_tau_0, max_tau_0=max_tau_0, step_tau_0=step_tau_0, epsilon=epsilon)
            print("Written {}".format(fn))


def two_dimensional_bifurcation_k_a(filename: str, T_0: float, t_0: float, n_f: int, tau_0: float, T_osc: float,
                                min_k: float = 0, max_k: float = 1, step_k: float = 0.005,
                                min_a: float = -1, max_a: float = 1, step_a: float = 0.005,
                                epsilon: float = 1e-2):
    # T_results: List[List[float]] = []
    # t_results: List[List[float]] = []
    k_values: List[float] = []
    a_values: List[float] = []

    k = min_k

    while k < max_k:
        k_values.append(k)
        k = k + step_k

    a = min_a

    while a < max_a:
        a_values.append(a)
        a += step_a

    periodicity_grid: np.ndarray = np.zeros(shape=(len(k_values), len(a_values), 1), dtype=np.uint16)

    white = 2**16 - 1
    gray_factor = math.floor(white / (n_f - 1))

    for i, k in enumerate(k_values):

        if not i % 10:
            print("Row progress: {}%".format(round(i / len(k_values) * 100, 2)))

        for j, a in enumerate(a_values):
            if not j % 20:
                print("Column progress: {}%".format(round(j / len(a_values) * 100, 2)))

            T_values, t_values = perform_simulation(T_0, t_0, n_f, a, k, T_osc, tau_0)

            periodicity = determine_periodicity(T_values, t_values, epsilon=epsilon)

            print("Found period {}".format(periodicity))

            if periodicity is None:
                periodicity = 0
            else:
                # periodicity = 255 - 2 * (periodicity - 1)
                periodicity = white - gray_factor * (periodicity - 1)

                if periodicity < 0:
                    print("ERROR!!!! NEGATIVE PERIODICITY {}!".format(periodicity))
                    periodicity = 0

            periodicity_grid[i, j] = periodicity

        # plt.imshow(periodicity_grid)
        # plt.show(block=False)

    # print("We have periodicity grid:")
    # print(periodicity_grid.tostring())

    # imshow(periodicity_grid, plugin='matplotlib')
    # plt.imshow(periodicity_grid)
    imsave(filename, arr=periodicity_grid)


def full_two_dimensional_bifurcation_stack_a_k(base_filename: str, T_0: float, t_0: float, n_f: int, T_osc: float,
                                               min_k: float = 0, max_k: float = 1, step_k: float = 0.005,
                                               min_tau_0: float = 0, max_tau_0: float = 30, step_tau_0: float = 0.005,
                                               epsilon: float = 1e-2, min_a: float = -1, max_a: float = 1,
                                               step_a: float = 0.01, reverse_a: bool = False):
    tau_0_values: List[float] = []

    tau_0 = min_tau_0

    while tau_0 < max_tau_0:
        tau_0_values.append(tau_0)
        tau_0 += step_tau_0

    if reverse_a:
        tau_0_values.reverse()

    for i, tau_0 in enumerate(tau_0_values):
        if not i % 5:
            print("Layer progress: {:0.2%}%".format(i/len(tau_0_values)))

        filename_a_string = "{:03d}_tau_{}".format(i, round(tau_0, 2))
        fn = base_filename.format(filename_a_string)
        two_dimensional_bifurcation_k_a(fn, T_0=T_0, t_0=t_0, n_f=n_f, tau_0=tau_0, T_osc=T_osc, min_k=min_k, max_k=max_k, step_k=step_k, min_a=min_a, max_a=max_a, step_a=step_a, epsilon=epsilon)
        print("Written {}".format(fn))


def two_dimensional_bifurcation_a_tau_0(filename: str, T_0: float, t_0: float, n_f: int, k: float, T_osc: float,
                                        min_tau_0: float = 0, max_tau_0: float = 5, step_tau_0: float = 0.005,
                                        min_a: float = -1, max_a: float = 1, step_a: float = 0.005,
                                        epsilon: float = 1e-2):
    # T_results: List[List[float]] = []
    # t_results: List[List[float]] = []
    tau_0_values: List[float] = []
    a_values: List[float] = []

    tau_0 = min_tau_0

    while tau_0 < max_tau_0:
        tau_0_values.append(tau_0)
        tau_0 = tau_0 + step_tau_0

    a = min_a

    while a < max_a:
        a_values.append(a)
        a += step_a

    periodicity_grid: np.ndarray = np.zeros(shape=(len(tau_0_values), len(a_values), 1), dtype=np.uint16)

    white = 2**16 - 1
    gray_factor = math.floor(white / (n_f - 1))

    for i, tau_0 in enumerate(tau_0_values):

        if not i % 10:
            print("Row progress: {}%".format(round(i / len(tau_0_values) * 100, 2)))

        for j, a in enumerate(a_values):
            if not j % 20:
                print("Column progress: {}%".format(round(j / len(a_values) * 100, 2)))

            T_values, t_values = perform_simulation(T_0, t_0, n_f, a, k, T_osc, tau_0)

            periodicity = determine_periodicity(T_values, t_values, epsilon=epsilon)

            print("Found period {}".format(periodicity))

            if periodicity is None:
                periodicity = 0
            else:
                # periodicity = 255 - 2 * (periodicity - 1)
                periodicity = white - gray_factor * (periodicity - 1)

                if periodicity < 0:
                    print("ERROR!!!! NEGATIVE PERIODICITY {}!".format(periodicity))
                    periodicity = 0

            periodicity_grid[i, j] = periodicity

        # plt.imshow(periodicity_grid)
        # plt.show(block=False)

    # print("We have periodicity grid:")
    # print(periodicity_grid.tostring())

    # imshow(periodicity_grid, plugin='matplotlib')
    # plt.imshow(periodicity_grid)
    imsave(filename, arr=periodicity_grid)


def __run_iterates_for_k_values(k_values_list: List[int], starting_index: int, base_filename: str, T_0: float,
                                t_0: float, n_f: int, T_osc: float,
                                min_tau_0: float, max_tau_0: float, step_tau_0: float,
                                min_a: float, max_a: float, step_a: float, epsilon: float):
    for i, k in enumerate(k_values_list):
        filename_a_string = "{:03d}_k_{}".format(i + starting_index, round(k, 2))
        fn = base_filename.format(filename_a_string)
        two_dimensional_bifurcation_a_tau_0(fn, T_0=T_0, t_0=t_0, n_f=n_f, k=k, T_osc=T_osc,
                                            min_tau_0=min_tau_0, max_tau_0=max_tau_0, step_tau_0=step_tau_0,
                                            min_a=min_a, max_a=max_a, step_a=step_a, epsilon=epsilon)
        print("Written {}".format(fn))


def full_two_dimensional_bifurcation_stack_a_tau_0(base_filename: str, T_0: float, t_0: float, n_f: int, T_osc: float,
                                                   min_k: float = 0, max_k: float = 1, step_k: float = 0.005,
                                                   min_tau_0: float = 0, max_tau_0: float = 30, step_tau_0: float = 0.005,
                                                   epsilon: float = 1e-2, min_a: float = -1, max_a: float = 1,
                                                   step_a: float = 0.01, reverse_k: bool = False, threaded: bool=True,
                                                   frames_per_thread: int = 10, use_processes: bool = True, num_cores: int = 4):
    k_values: List[float] = []

    k = min_k

    while k < max_k:
        k_values.append(k)
        k += step_k

    if reverse_k:
        k_values.reverse()

    if use_processes:

        frames_per_core = int(ceil(len(k_values)/num_cores))

        for j in range(0, len(k_values), frames_per_core):
            k_sublist = k_values[j:j+frames_per_core]

            Process(target=__run_iterates_for_k_values, kwargs={
                "k_values_list": k_sublist,
                "starting_index": j,
                "base_filename": base_filename,
                "T_0": T_0,
                "t_0": t_0,
                "n_f": n_f,
                "T_osc": T_osc,
                "min_tau_0": min_tau_0,
                "max_tau_0": max_tau_0,
                "step_tau_0": step_tau_0,
                "min_a": min_a,
                "max_a": max_a,
                "step_a": step_a,
                "epsilon": epsilon
            }).start()

        return

    if not threaded:
        for i, k in enumerate(k_values):
            if not i % 5:
                print("Layer progress: {:0.2%}%".format(i/len(k_values)))

            filename_a_string = "{:03d}_k_{}".format(i, round(k, 2))
            fn = base_filename.format(filename_a_string)
            two_dimensional_bifurcation_a_tau_0(fn, T_0=T_0, t_0=t_0, n_f=n_f, k=k, T_osc=T_osc, min_tau_0=min_tau_0, max_tau_0=max_tau_0, step_tau_0=step_tau_0, min_a=min_a, max_a=max_a, step_a=step_a, epsilon=epsilon)
            print("Written {}".format(fn))

    else:

        def _run_iterates_for_k_values(k_values_list: List[int], starting_index:int):
            for i, k in enumerate(k_values_list):
                filename_a_string = "{:03d}_k_{}".format(i+starting_index, round(k, 2))
                fn = base_filename.format(filename_a_string)
                two_dimensional_bifurcation_a_tau_0(fn, T_0=T_0, t_0=t_0, n_f=n_f, k=k, T_osc=T_osc,
                                                    min_tau_0=min_tau_0, max_tau_0=max_tau_0, step_tau_0=step_tau_0,
                                                    min_a=min_a, max_a=max_a, step_a=step_a, epsilon=epsilon)
                print("Written {}".format(fn))

        for j in range(0, len(k_values), frames_per_thread):
            k_sublist = k_values[j:j+frames_per_thread]

            Thread(target=_run_iterates_for_k_values, kwargs={
                "k_values_list": k_sublist,
                "starting_index": j
            }).start()
