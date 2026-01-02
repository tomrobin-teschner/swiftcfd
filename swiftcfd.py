import swiftcfd

def run():
    # create parameters
    params = swiftcfd.parameters()
    params.read_from_file('input/heatedCavity.toml')

    # create mesh
    mesh = swiftcfd.mesh(params)
    mesh.create()

    # add to field manager
    field_manager = swiftcfd.field_manager(mesh)
    field_manager.add_field('T')

    # create governign equation
    eqn = swiftcfd.heat_diffusion(params, mesh, 'T')

    # create time handler
    time = swiftcfd.time(params, mesh, eqn)

    # create output
    out = swiftcfd.output(params, mesh, field_manager)

    # create performance statistics
    stats = swiftcfd.performance_statistics()
    stats.timer_start()

    # loop over time
    iter = 1
    while (time.not_reached_end_time()):
        # copy solution
        field_manager.update()

        # compute time step
        dt, CFL = time.compute_dt(field_manager.fields['T'])

        # update equation for current timestep
        eqn.update(time, field_manager.fields['T'])

        # update time steps
        time.current_time += dt

        # convergence checking
        is_diagonal, num_iterations, res_norm, has_converged = eqn.solver.get_solver_statistics()

        # update statistics
        stats.add_timestep_statistics(eqn)

        if is_diagonal:
            print(f'Time: {time.current_time:.2e}, dt: {dt:.1f}, CFL: {CFL:.2f}')
        else:
            print(f'Time: {time.current_time:.2e}, dt: {dt:.1f}, CFL: {CFL:.2f}, iterations: {num_iterations:<5}, res_norm: {res_norm:.3e}, has_converged: {has_converged}')
        # save solution animation
        out.write(iter)
        iter += 1

    # print statistics to console
    stats.timer_end()
    stats.print_statistics()

    
if __name__ == '__main__':
    run()