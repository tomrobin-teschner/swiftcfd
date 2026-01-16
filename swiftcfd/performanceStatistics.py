from time import perf_counter as clock

import numpy as np

class PerformanceStatistics:
    def __init__(self, equations):
        self.time_start = 0
        self.time_end = 0
        self.solver_time = 0
        
        # create dictionary that holds iterations for each iteration and each variable
        var_names = [eq.get_variable_name() for eq in equations]
        self.iterations = {}
        for var in var_names:
            self.iterations[var] = []
    
    def timer_start(self):
        self.timer_start = clock()
    
    def timer_end(self):
        self.timer_end = clock()
        self.solver_time = self.timer_end - self.timer_start

    def add_timestep_statistics(self, equation):
        is_diagonal, num_iterations, res_norm, has_converged = equation.solver.get_solver_statistics()
        self.iterations[equation.get_variable_name()].append(num_iterations)

    def print_statistics(self):
        print(f'\nSimulation finished in {self.solver_time:.2f} seconds.')
        print(f'Total number of time steps: {len(next(iter(self.iterations.values())))}.')
        for key, value in self.iterations.items():
            print(f'Average number of iterations for {key}: {np.mean(value):.2f}')
        print()