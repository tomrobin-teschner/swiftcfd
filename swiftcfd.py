import swiftcfd

from swiftcfd.equations.equations.primitiveVariables import PrimitiveVariables as pv

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

    # create runtime handler
    runtime = swiftcfd.runtime(params, mesh, field_manager, equations)

    # create output
    output = swiftcfd.output(params, mesh, field_manager)

    # create performance statistics
    stats = swiftcfd.performance_statistics(equations)
    stats.timer_start()

    # logger class to print output to console
    log = swiftcfd.log()

    # loop over time
    while (runtime.not_reached_end_time()):
        # copy solution
        field_manager.update_solution()

        # compute time step
        runtime.compute_dt()

        # linearisation step through picard iterations
        while(runtime.not_reached_final_picard_iteration()):
            # update picard solution
            field_manager.update_picard_solution()

            # perform any pre-solve tasks
            for eqn in equations:
                eqn.pre_solve_task(runtime)

            # solve equations
            for eqn in equations:
                # update equation for current timestep
                eqn.solve(runtime)
                
                # update statistics
                stats.add_timestep_statistics(eqn)
            
            # perform any post-solve tasks
            for eqn in equations:
                eqn.post_solve_task(runtime)

            # print time step statistics
            log.print_time_step(runtime, equations)
        
        # update time steps
        runtime.update_time()

        # save solution animation
        if params('solver', 'output', 'writingFrequency') > 0 and runtime.timestep % params('solver', 'output', 'writingFrequency') == 0:
            output.write(runtime.timestep)

    # print statistics to console
    stats.timer_end()
    stats.print_statistics()

    # write solution
    output.write()

if __name__ == '__main__':
    run()