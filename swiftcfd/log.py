class Log:
    def __init__(self):
        pass

    def print_time_step(self, time, equations):
        current_time = time.current_time
        cfl = time.CFL
        dt = time.dt

        print(f'Time step: {time.timestep + 1:<5}, Current time: {time.current_time:.2e}, CFL: {cfl:.2e}, dt: {dt:.2e}')

        for equation in equations:
            is_diagonal, num_iterations, res_norm, has_converged = equation.solver.get_solver_statistics()

            if not is_diagonal:
                print(f'{equation.get_variable_name()}: iterations: {num_iterations:<5}, Residual: {res_norm:.3e}, Converged?: {has_converged}')
        print()