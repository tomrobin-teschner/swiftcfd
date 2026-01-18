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
    stats = swiftcfd.performance_statistics(params, equations)
    stats.timer_start()

    # logger class to print output to console
    log = swiftcfd.log()

    # create residual calculating object
    residuals = swiftcfd.residuals(params, field_manager)

    # loop over time
    while (runtime.has_not_reached_final_time()):
        # copy solution
        field_manager.update_solution()

        # compute time step
        runtime.compute_dt()

        # print time info to console
        log.print_time_info(runtime)

        # linearisation step through picard iterations
        while(runtime.has_not_reached_final_picard_iteration()):
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
            
            # compute picard residuals
            has_converged = residuals.check_picard_convergence(runtime)

            # print time step statistics
            log.print_picard_iteration(runtime, equations, residuals)

            if has_converged:
                break
        
        # update time steps
        runtime.update_time()

        # cehck for simulation convergence
        has_converged = residuals.check_convergence(runtime)

        # print convergence information for current time step
        log.print_convergence_info(runtime, equations, residuals)

        # save solution animation
        if params('solver', 'output', 'writingFrequency') > 0 and runtime.timestep % params('solver', 'output', 'writingFrequency') == 0:
            output.write(runtime.timestep)

        if has_converged:
            break

    # print statistics to console
    stats.timer_end()
    stats.write_statistics()

    # write solution
    output.write()

    # write residuals
    residuals.write()

if __name__ == '__main__':
    run()