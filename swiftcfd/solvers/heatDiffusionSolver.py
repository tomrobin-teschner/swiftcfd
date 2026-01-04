import swiftcfd

def run(filename):
    # create parameters
    params = swiftcfd.parameters()
    params.read_from_file(filename)

    # create mesh
    mesh = swiftcfd.mesh(params)
    mesh.create()

    # create governign equation
    TEqn = swiftcfd.equation_factory(params, mesh).create('heatDiffusion')

    # add to field manager
    field_manager = swiftcfd.field_manager(mesh)
    field_manager.add_field(TEqn.var_name)

    # create time handler
    time = swiftcfd.time(params, mesh, TEqn)

    # create output
    output = swiftcfd.output(params, mesh, field_manager)

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
        time.compute_dt(field_manager.fields[TEqn.var_name])

        # update equation for current timestep
        TEqn.update(time, field_manager.fields[TEqn.var_name])

        # update time steps
        time.update_time()

        # convergence checking
        is_diagonal, num_iterations, res_norm, has_converged = TEqn.solver.get_solver_statistics()

        # update statistics
        stats.add_timestep_statistics(TEqn)

        # print time step statistics
        log.print_time_step(time, TEqn)
        
        # save solution animation
        if params('solver', 'output', 'writingFrequency') > 0 and time.timestep % params('solver', 'output', 'writingFrequency') == 0:
            output.write(time.timestep)

    # print statistics to console
    stats.timer_end()
    stats.print_statistics()

    # write solution
    output.write()