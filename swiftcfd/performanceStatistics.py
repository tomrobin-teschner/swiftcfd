from time import perf_counter as clock

import numpy as np

class PerformanceStatistics:
    def __init__(self):
        self.time_start = 0
        self.time_end = 0
        self.solver_time = 0

        self.iterations = []
    
    def timer_start(self):
        self.timer_start = clock()
    
    def timer_end(self):
        self.timer_end = clock()
        self.solver_time = self.timer_end - self.timer_start

    def add_timestep_statistics(self, equation):
        is_diagonal, num_iterations, res_norm, has_converged = equation.solver.get_solver_statistics()
        self.iterations.append(num_iterations)

    def print_statistics(self):
        print(f'\nSimulation finished in {self.solver_time:.2f} seconds.')
        print(f'Average number of iterations: {np.mean(self.iterations):.2f}')
        print(f'Total number of time steps: {len(self.iterations)}\n')