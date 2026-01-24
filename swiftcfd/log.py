from blessed import Terminal

class Log:
    def __init__(self):
        self.term = Terminal()
        print(self.term.home + self.term.clear, end='')
    
    def print_time_info(self, runtime):
        current_time = runtime.current_time
        runtime.compute_CFL()
        cfl = runtime.CFL
        dt = runtime.dt

        print(self.term.move_xy(0,  0) + f'Time step: {runtime.timestep + 1:<5}')
        print(self.term.move_xy(20, 0) + f'Current time: {runtime.current_time:.2e}')
        print(self.term.move_xy(50, 0) + f'CFL: {cfl:.2e}')
        print(self.term.move_xy(70, 0) + f'dt: {dt:.2e}')

    def print_picard_iteration(self, runtime, equations, residuals):
        print(self.term.move_xy(0, 1) + f'Picard iteration: {runtime.current_picard_iteration:>4}/{runtime.num_picard_iterations}')

        row = 2
        for equation in equations:
            is_diagonal, num_iterations, res_norm, has_converged = equation.solver.get_solver_statistics()

            if not is_diagonal:
                if equation.requires_linearisation:
                    print(self.term.move_xy(0, row) + f'{equation.get_variable_name()}: Picard residual: {residuals.picard_current_residual[equation.get_variable_name()]:.2e}, iterations: {num_iterations:<5}, Residual: {res_norm:.3e}')
                else:
                    print(self.term.move_xy(0, row) + f'{equation.get_variable_name()}: Iterations: {num_iterations:<5}, Residual: {res_norm:.3e}')
                row += 1
    
    def print_convergence_info(self, runtime, equations, residuals):
        row = 3 + len(equations)
        col = 0
        for var_name, residuals in residuals.convergence_residual.items():
            print(self.term.move_xy(col, row) + f'{var_name}: {residuals[-1]:.2e}')
            col += 20