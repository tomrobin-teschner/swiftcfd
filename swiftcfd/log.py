from blessed import Terminal

class Log:
    def __init__(self):
        self.term = Terminal()
        print(self.term.home + self.term.clear, end='')

    def print_time_step(self, runtime, equations):
        print(self.term.home + self.term.clear, end='')

        current_time = runtime.current_time
        cfl = runtime.CFL
        dt = runtime.dt

        # if runtime.current_picard_iteration == 1:
        print(f'Time step: {runtime.timestep + 1:<5}', end='')
        print(self.term.move_xy(20, 0) + f'Current time: {runtime.current_time:.2e}', end='')
        print(self.term.move_xy(50, 0) + f'CFL: {cfl:.2e}', end='')
        print(self.term.move_xy(70, 0) + f'dt: {dt:.2e}')


        print(self.term.move_xy(0, 1) + f'Picard iteration: {runtime.current_picard_iteration}/{runtime.num_picard_iterations}')

        row = 2
        for equation in equations:
            is_diagonal, num_iterations, res_norm, has_converged = equation.solver.get_solver_statistics()

            if not is_diagonal:
                print(self.term.move_xy(0, row) + f'{equation.get_variable_name()}: iterations: {num_iterations:<5}, Residual: {res_norm:.3e}, Converged?: {has_converged}')
                row += 1