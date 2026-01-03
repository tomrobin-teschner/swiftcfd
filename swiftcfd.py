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

    # logger class to print output to console
    log = swiftcfd.log()

    # loop over time
    while (time.not_reached_end_time()):
        # copy solution
        field_manager.update()

        # compute time step
        dt, CFL = time.compute_dt(field_manager.fields['T'])

        # update equation for current timestep
        eqn.update(time, field_manager.fields['T'])

        # update time steps
        time.update_time()

        # convergence checking
        is_diagonal, num_iterations, res_norm, has_converged = eqn.solver.get_solver_statistics()

        # update statistics
        stats.add_timestep_statistics(eqn)

        # print time step statistics
        log.print_time_step(time, eqn)

        
        
        # # save solution animation
        # out.write(time.timestep)

    # print statistics to console
    stats.timer_end()
    stats.print_statistics()

    # write solution
    out.write()

    
if __name__ == '__main__':
    run()