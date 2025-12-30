import fluxy

def run():
    # create parameters
    params = fluxy.parameters()

    # create mesh
    mesh = fluxy.mesh(params)
    mesh.create()

    # add to field manager
    field_manager = fluxy.field_manager(params, mesh)
    field_manager.add_field('T')

    # create governign equation
    eqn = fluxy.heat_diffusion(params, mesh, 'T')

    # create time handler
    time = fluxy.time(params, mesh, eqn.has_advection, eqn.has_diffusion)

    # create output
    out = fluxy.output(params, mesh, field_manager)

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