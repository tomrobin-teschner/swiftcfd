import fluxy

def main():
    # create parameters
    params = fluxy.parameters()

    # create mesh
    mesh = fluxy.mesh(params)
    mesh.create()

    # create temperature field
    Tn0 = fluxy.field(params, mesh, 'T')
    Tn1 = Tn0

    # create time handler
    time = fluxy.time(params, mesh)

    # loop over time
    while (time.reached_end_time() == False):
        # copy solution
        Tn0 = Tn1

        # compute time step
        dt, CFL = time.compute_dt()

        # update time steps
        time.current_time += dt

    # create output
    out = fluxy.output(params, mesh)
    out.register(Tn1, 'Temperature')

    out.write()
    out.write(132)

if __name__ == '__main__':
    main()