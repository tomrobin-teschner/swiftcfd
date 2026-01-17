from os.path import join
from time import perf_counter as clock

import numpy as np

class PerformanceStatistics:
    def __init__(self, params, equations):
        self.params = params
        self.time_start = 0
        self.time_end = 0
        self.solver_time = 0

        self.case = self.params('solver', 'output', 'filename')
        self.out_folder = join('output', self.case)
        
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

    def write_statistics(self):
        with open(join(self.out_folder, self.case + '_statistics.dat'), 'w') as f:
            f.write(f'Simulation time [s]:                {self.solver_time:.2f}\n')
            f.write(f'Total number of time steps:         {len(next(iter(self.iterations.values())))}\n')
            for key, value in self.iterations.items():
                f.write(f'Average number of iterations for {key}: {np.mean(value):.1f}\n')