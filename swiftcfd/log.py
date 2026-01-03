class Log:
    def __init__(self):
        pass

    def print_time_step(self, time, equation):
        current_time = time.current_time
        cfl = time.CFL
        dt = time.dt
        is_diagonal, num_iterations, res_norm, has_converged = equation.solver.get_solver_statistics()

        if is_diagonal:
            print(f'Time: {time.current_time:.2e}, dt: {dt:.1f}, CFL: {cfl:.2f}')
        else:
            print(f'Time: {time.current_time:.2e}, dt: {dt:.1f}, CFL: {cfl:.2f}, iterations: {num_iterations:<5}, res_norm: {res_norm:.3e}, has_converged: {has_converged}')