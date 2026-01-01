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
    time = swiftcfd.time(params, mesh, eqn.has_first_order_space_derivative, eqn.has_second_order_space_derivative)

    # create output
    out = swiftcfd.output(params, mesh, field_manager)

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
        num_iterations, res_norm, has_converged = eqn.solver.get_solver_statistics()

        print(f'Time: {time.current_time:<8}, dt: {dt:.1f}, CFL: {CFL:.2f}, iterations: {num_iterations:<5}, res_norm: {res_norm:.3e}, has_converged: {has_converged}')
        
        # save solution animation
        out.write(iter)
        iter += 1
    
    # report max temperature value
    print(f'\nMax temperature value: {field_manager.fields['T']._data.max():.2f}\n')

if __name__ == '__main__':
    run()