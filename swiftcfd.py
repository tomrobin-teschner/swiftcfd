import swiftcfd

def run():
    # read command line arguments
    cla_parser = swiftcfd.command_line_argument_parser()

    # create parameters
    params = swiftcfd.parameters()
    params.read_from_file(cla_parser.arguments.input)

    # create mesh
    mesh = swiftcfd.mesh(params)
    mesh.create()

    # create governign equation
    equations, field_manager = swiftcfd.equation_factory(params, mesh).create()

    # create time handler
    time = swiftcfd.time(params, mesh, field_manager, equations)

    # create output
    output = swiftcfd.output(params, mesh, field_manager)

    # create performance statistics
    stats = swiftcfd.performance_statistics(equations)
    stats.timer_start()

    # logger class to print output to console
    log = swiftcfd.log()

    # loop over time
    while (time.not_reached_end_time()):
        # copy solution
        field_manager.update()

        # compute time step
        time.compute_dt()

        for eqn in equations:
            # update equation for current timestep
            eqn.solve(time, field_manager.fields[eqn.var_name])
            
            # convergence checking
            is_diagonal, num_iterations, res_norm, has_converged = eqn.solver.get_solver_statistics()

            # update statistics
            stats.add_timestep_statistics(eqn)

            # # print time step statistics
            log.print_time_step(time, eqn)
        
        # create a new line
        print('')

        # update time steps
        time.update_time()

        # save solution animation
        if params('solver', 'output', 'writingFrequency') > 0 and time.timestep % params('solver', 'output', 'writingFrequency') == 0:
            output.write(time.timestep)

    # print statistics to console
    stats.timer_end()
    stats.print_statistics()

    # write solution
    output.write()

if __name__ == '__main__':
    run()