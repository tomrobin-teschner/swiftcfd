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

    # create output
    out = fluxy.output(params, mesh)
    out.register(Tn1, 'Temperature')
    out.write()
    out.write(132)

if __name__ == '__main__':
    main()