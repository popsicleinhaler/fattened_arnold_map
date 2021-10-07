import arnold_map

from numpy import pi, e

if __name__ == '__main__':
    # arnold_map.plot_simulation(0, 100, 1.60001, 0.25, 0.8, 24)
    # arnold_map.plot_simulation(t_0=0, n_f=1000, T_0=1, a=0.4, k=0.8, T_osc=24, epsilon=1e-2)
    # arnold_map.plot_simulation(t_0=1, n_f=1000, T_0=1, a=0.4, k=0.5, T_osc=24, epsilon=5e-2, tau_0=10)
    # arnold_map.plot_2d_phase_portrait(a=-1, k=1, T_osc=2*pi, t_0_max=5, T_0_max=10, t_0_step=0.5, T_0_step=0.5, tau_0=5)
    # arnold_map.full_two_dimensional_bifurcation_stack_a_k(
    #     base_filename="periodicity_grid_a_k_2/periodicity_grid_{}.png", t_0=0, T_0=1, T_osc=3, epsilon=0.05,
    #     max_tau_0=3.5, step_tau_0=0.01, step_a=0.01, max_a=1.09, n_f=1000)
    # arnold_map.two_dimensional_bifurcation("larger_figure.png", T_osc=3, T_0=2, t_0=0, a=0, n_f=1000, max_k=2, step_k=0.01, step_tau_0=0.01, max_tau_0=3.5)
    # arnold_map.two_dimensional_bifurcation("even_larger_figure.png", T_osc=3, T_0=2, t_0=0, a=0, n_f=1000, max_k=4, step_k=0.01, step_tau_0=0.01, max_tau_0=3.5)
    # arnold_map.two_dimensional_bifurcation("even_larger_figure_really_hi_res.png", T_osc=3, T_0=2, t_0=0, a=0, n_f=1000, max_k=4, step_k=0.01/4, step_tau_0=0.01/4, max_tau_0=3.5)
    # arnold_map.two_dimensional_bifurcation("irrational_initial_conditions.png", T_osc=pi + 1e-4, T_0=2/e, t_0=1/4 * pi + 1/9 * e, a=0, n_f=1000, max_k=4, step_k=0.01 + 0.001*pi, step_tau_0=0.01 + 0.001*pi, max_tau_0=3.5)
    # arnold_map.plot_2d_phase_portrait(a=0, )
    # arnold_map.two_dimensional_bifurcation_k_a(filename="two_dimensional_art.png", tau_0=0.63, t_0=0, T_0=1, T_osc=3, epsilon=0.05, step_a=0.0025, max_a=1.09, n_f=1000, step_k=0.00125)
    # arnold_map.full_two_dimensional_bifurcation_stack_a_k(base_filename="periodicity_grid_a_k/periodicity_grid_{}.png", t_0=0, T_0=4, T_osc=3, epsilon=0.05, max_tau_0=3.5, step_tau_0=0.01, step_a=0.01, max_a=1.09, n_f=1000)
    # arnold_map.full_two_dimensional_bifurcation_stack_a_tau_0(base_filename="periodicity_grid_a_tau_0/periodicity_grid_{}.png", t_0=0, T_0=2, T_osc=3, epsilon=0.05, max_tau_0=3.5, step_tau_0=0.01, step_a=0.01, max_a=1.09, n_f=1000, threaded=False)
    # arnold_map.full_two_dimensional_bifurcation_stack_a_tau_0(base_filename="periodicity_grid_a_tau_0_threaded/periodicity_grid_{}.png", t_0=0.0001*pi, T_0=2.3342334 + 0.001*pi, T_osc=3, epsilon=0.05, max_tau_0=3.5, step_tau_0=0.01, step_a=0.01, max_a=1.09, n_f=1000, threaded=True)
    # arnold_map.full_two_dimensional_bifurcation_stack_a_tau_0(base_filename="periodicity_grid_a_tau_0_multiprocess/periodicity_grid_{}.png", t_0=0.0001*pi, T_0=2.3342334 + 0.001*pi, T_osc=3, epsilon=0.05, max_tau_0=3.5, step_tau_0=0.01, step_a=0.01, max_a=1.09, n_f=1000, use_processes=True, num_cores=6)
    # arnold_map.two_dimensional_bifurcation_a_tau_0(filename="periodicity_grid_a_tau_0_large.png", t_0=0.0001*pi, T_0=2.3342334 + 0.001*pi, T_osc=3, epsilon=0.05, max_tau_0=3.5, step_tau_0=0.01/2, step_a=0.01/2, max_a=1.09, n_f=1000, k=0.95)
    # arnold_map.two_dimensional_bifurcation_a_tau_0(filename="periodicity_grid_a_tau_0_large_k_1.png", t_0=0.0001*pi, T_0=2.3342334 + 0.001*pi, T_osc=3, epsilon=0.05, max_tau_0=3.5, step_tau_0=0.01/2, step_a=0.01/2, max_a=1.09, n_f=1000, k=1)
    # arnold_map.two_dimensional_bifurcation_a_tau_0(filename="periodicity_grid_a_tau_0_larger_k_1.png", t_0=0.0001*pi, T_0=2.3342334 + 0.001*pi, T_osc=3, epsilon=0.05, max_tau_0=3.5, step_tau_0=0.01/4, step_a=0.01/4, max_a=1.09, n_f=1000, k=1)
    # arnold_map.two_dimensional_bifurcation_a_tau_0(filename="periodicity_grid_a_tau_0_k_0.19.png", t_0=0.0001*pi, T_0=2.3342334 + 0.001*pi, T_osc=3, epsilon=0.05, max_tau_0=3.5, step_tau_0=0.01/2, step_a=0.01/2, max_a=1.09, n_f=1000, k=0.19)
    # arnold_map.two_dimensional_bifurcation_a_tau_0(filename="periodicity_grid_a_tau_0_k_0.19.png", t_0=0, T_0=2, T_osc=3, epsilon=0.05, max_tau_0=3.5, step_tau_0=0.01, step_a=0.01, max_a=1.09, n_f=1000, k=0.19)
    # arnold_map.two_dimensional_bifurcation_a_tau_0(filename="periodicity_grid_a_tau_0_k_0.63.png", t_0=0, T_0=2, T_osc=3, epsilon=0.05, max_tau_0=3.5, step_tau_0=0.01, step_a=0.01, max_a=1.09, n_f=1000, k=0.63)
    arnold_map.two_dimensional_bifurcation_a_tau_0(filename="periodicity_grid_a_tau_0_k_1.0.png", t_0=0, T_0=2, T_osc=3, epsilon=0.05, max_tau_0=3.5, step_tau_0=0.01, step_a=0.01, max_a=1.09, n_f=1000, k=1.0)

    # arnold_map.two_dimensional_bifurcation("test_2d_bif_3.png", t_0=1, n_f=1000, a=1, T_osc=2*pi, max_T_0=5, epsilon=5e-2)
    # arnold_map.full_two_dimensional_bifurcation_stack(base_filename="new_periodicity_grids/periodicity_grid_large_{"
    #                                                                 "}.png", T_osc=2*pi, t_0=1, n_f=1000, max_tau_0=5,
    #                                                   epsilon=5e-2, step_a=0.05, step_k=5e-3/3, min_a=0, max_a=1,
    #                                                   reverse_a=True)
    # arnold_map.plot_simulation(t_0=0, n_f=1000, T_0=1, a=0.4, k=0.8, T_osc=24, epsilon=0.05, tau_0=4.85)
    # arnold_map.plot_simulation(t_0=0, n_f=1000, T_0=12, a=0.4, k=0.8, T_osc=24, epsilon=0.05, tau_0=12)
    # arnold_map.plot_2d_phase(t_0=0, n_f=1000, T_0=12, a=0.4, k=0.8, T_osc=24, tau_0=12)
    # arnold_map.plot_simulation(t_0=0, n_f=50000, T_0=1, a=0.4, k=0.8, T_osc=24, tau_0=1.3350, epsilon=0.05)
    # arnold_map.plot_2d_phase(t_0=0, n_f=1000, T_0=6, a=1, k=0.8, T_osc=2*pi)
    # arnold_map.plot_simulation(t_0=0, n_f=1000, T_0=4.85, a=0.4, k=0.8, T_osc=24, epsilon=1e-2)
    # arnold_map.plot_simulation(a=1, t_0=8, T_0=12, T_osc=3, n_f=1000, k=0.3, epsilon=0.005, tau_0=36)
    # arnold_map.two_dimensional_bifurcation(filename="periodicity_grid11.png", t_0=0, T_0=4, T_osc=3, n_f=1000, a=0.5, epsilon=0.05, max_tau_0=10, step_tau_0=0.01)
    # arnold_map.full_two_dimensional_bifurcation_stack(base_filename="fixed_periodicity_grids/periodicity_grid_{}.png", t_0=0, T_0=4, T_osc=3, epsilon=0.05, max_tau_0=10, step_tau_0=0.01, step_a=0.1, max_a=1.09, n_f=1000)
    # arnold_map.full_two_dimensional_bifurcation_stack(base_filename="periodicity_grids_a_0.9_1.0/periodicity_grid_{}.png", t_0=0, T_0=4, T_osc=3, epsilon=0.05, max_tau_0=10, step_tau_0=0.01, step_a=0.005, min_a=0.9, max_a=1.04, n_f=1000, use_processes=True, num_cores=6)
    # arnold_map.full_two_dimensional_bifurcation_stack(base_filename="periodicity_grids_a_0.995_1.0/periodicity_grid_{}.png", t_0=0, T_0=4, T_osc=3, epsilon=0.05, max_tau_0=10, step_tau_0=0.01, step_a=0.00025, min_a=0.995, max_a=1.001, n_f=1000, use_processes=True, num_cores=6)
    # arnold_map.full_two_dimensional_bifurcation_stack(base_filename="periodicity_grids_a_0.999_1.0/periodicity_grid_{}.png", t_0=0, T_0=4, T_osc=3, epsilon=0.05, max_tau_0=10, step_tau_0=0.01, step_a=0.0001, min_a=0.999, max_a=1.001, n_f=1000, use_processes=True, num_cores=6)
    arnold_map.full_two_dimensional_bifurcation_stack(base_filename="periodicity_grids_a_0.9999_1.0/periodicity_grid_{}.png", t_0=0, T_0=4, T_osc=3, epsilon=0.05, max_tau_0=10, step_tau_0=0.01, step_a=0.000005, min_a=0.9999, max_a=1.0001, n_f=1000, use_processes=True, num_cores=6)
    # arnold_map.two_dimensional_bifurcation("fixed_periodicity_grids/periodicity_grid_{}_adjusted.png", t_0=0, T_0=4, T_osc=3, epsilon=0.05, max_tau_0=10, step_tau_0=0.01, a=1, n_f=1000)
    # arnold_map.two_dimensional_bifurcation("fixed_periodicity_grids/periodicity_grid_a_0_adjusted_n1000.png", t_0=0,
    #                                        T_0=4,
    #                                        T_osc=3, epsilon=0.05, max_tau_0=10, step_tau_0=0.01, a=0, n_f=1000)
    # arnold_map.two_dimensional_bifurcation("fixed_periodicity_grids/periodicity_grid_a_0.5_zoomed.png", t_0=0, T_0=4,
    #                                        T_osc=3, epsilon=0.05, max_tau_0=3.25, step_tau_0=0.0025, step_k=0.005/4, a=0.5, n_f=1000)
    # arnold_map.plot_2d_phase(t_0=1, n_f=100, T_0=1, a=0.4, k=0.5, T_osc=24, tau_0=10)
    # arnold_map.plot_simulation(t_0=1, n_f=100, T_0=1, a=0.4, k=0.5, T_osc=24, epsilon=5e-2, tau_0=10)
    # arnold_map.full_two_dimensional_bifurcation_stack(base_filename="periodicity_grids_24_hours/periodicity_grid_{}.png",
    #                                                   t_0=0, T_0=4, T_osc=24, epsilon=0.05, max_tau_0=26,
    #                                                   step_tau_0=0.01, step_a=0.1, max_a=0.09, n_f=1000)
    # arnold_map.full_two_dimensional_bifurcation_stack(
    #     base_filename="periodicity_grids_24_hours/periodicity_grid_{}.png",
    #     t_0=0, T_0=4, T_osc=24, epsilon=0.05, max_tau_0=26,
    #     step_tau_0=0.01, step_a=0.1, min_a=0.0, max_a=1.09, n_f=1000)
    # arnold_map.full_two_dimensional_bifurcation_stack(base_filename="fixed_periodicity_grids/periodicity_grid_{}.png",
    #                                                   t_0=0, T_0=4, T_osc=3, epsilon=0.05, max_tau_0=10,
    #                                                   step_tau_0=0.01, step_a=0.1, min_a=-0.95, max_a=1.00, n_f=1000)
    # arnold_map.two_dimensional_bifurcation_T_0_k(filename="periodicity_grid3.png", t_0=0, n_f=1000, a=0.5, T_osc=24)
    # arnold_map.full_two_dimensional_bifurcation_stack(base_filename="periodicity_grids/periodicity_grid_{}.png",
    #                                                   T_osc=24, n_f=1000, t_0=0, step_a=0.1, max_T_0=26)
    # arnold_map.plot_a_bifurcations(T_0=5, t_0=0, n_f=1000, k=0.8, T_osc=24, step_a=0.001, num_iterates_for_diagram=20, tau_0=3)
    # arnold_map.plot_a_bifurcations(T_0=5, t_0=0, n_f=1000, k=0, T_osc=24, step_a=0.001, num_iterates_for_diagram=20, tau_0=3)
    # arnold_map.plot_a_bifurcations(T_0=5, t_0=0, n_f=1000, k=1, T_osc=24, step_a=0.001, num_iterates_for_diagram=20, tau_0=3)
    # arnold_map.plot_tau_0_bifurcations(T_0=5, t_0=0, n_f=1000, k=0.8, T_osc=24, step_tau_0=0.05, num_iterates_for_diagram=30, a=0.5)
    # arnold_map.plot_tau_0_bifurcations(T_0=5, t_0=0, n_f=1000, k=0.8, T_osc=24, step_tau_0=0.05, a=1)
    # arnold_map.plot_tau_0_bifurcations(T_0=5, t_0=0, n_f=1000, k=0.8, T_osc=24, step_tau_0=0.05, a=0.5)
    # arnold_map.plot_tau_0_bifurcations(T_0=5, t_0=0, n_f=1000, k=0.8, T_osc=24, step_tau_0=0.05, a=0)
    # arnold_map.plot_tau_0_bifurcations(T_0=5, t_0=0, n_f=1000, k=0.8, T_osc=24, step_tau_0=0.05, a=-0.5)
    # arnold_map.plot_tau_0_bifurcations(T_0=5, t_0=0, n_f=1000, k=0, T_osc=24, step_tau_0=0.05, a=0.5)
    # arnold_map.plot_tau_0_bifurcations(T_0=5, t_0=0, n_f=1000, k=1, T_osc=24, step_tau_0=0.05, a=0.5)
    # arnold_map.plot_tau_0_bifurcations(T_0=5, t_0=0, n_f=1000, k=0.8, T_osc=24, step_tau_0=0.05, a=)
    # arnold_map.plot_tau_0_bifurcations(T_0=5.24325, t_0=4.12213123, n_f=1000, k=0.8, T_osc=24, step_tau_0=0.01, a=0.8)
    # arnold_map.plot_k_bifurcations(T_0=5, t_0=0, n_f=1000, tau_0=3, T_osc=24, num_iterates_for_diagram=30, a=0.5, step_k=0.005)
    # arnold_map.plot_k_bifurcations(T_0=2, t_0=0, n_f=1000, tau_0=3, T_osc=24, a=1, step_k=0.005)
    # arnold_map.plot_k_bifurcations(T_0=2, t_0=0, n_f=1000, tau_0=3, T_osc=24, a=0, step_k=0.005)
    # arnold_map.plot_k_bifurcations(T_0=2, t_0=0, n_f=1000, tau_0=3, T_osc=24, a=-1, step_k=0.005)
    # arnold_map.two_dimensional_bifurcation(t_0=0, T_0=4, T_osc=3, epsilon=0.05, max_tau_0=10, step_tau_0=0.01, a=0.7, n_f=1000)
    # arnold_map.plot_2d_phase(t_0=0.23, )
    # arnold_map.plot_simulation(t_0=0.23, T_0=4, T_osc=24, n_f=1000, tau_0=20, k=0.6, a=1, epsilon=1e-1)
    # arnold_map.plot_2d_phase_portrait(T_osc=24, n_f=1000, tau_0=20, k=0.6, a=1, t_0_step=1, T_0_step=1, t_0_max=30, T_0_max=30)

    # arnold_map.plot_T_0_bifurcations(t_0=0, n_f=1000, a=0.4,
    #                                  k=0.8, T_osc=24, step_T_0=0.005, num_iterates_for_diagram=40)
    # arnold_map.plot_k_bifurcations(0, 1000, 2, 0.4, 24, step_k=0.005, num_iterates_for_diagram=30)

    # arnold_map.logistic_map_bifurcations(x_0=0.2, r_step=0.001)
