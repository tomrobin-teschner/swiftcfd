import fluxy

def main():
    # create parameters
    params = fluxy.parameters()

    # create mesh
    mesh = fluxy.mesh(params)
    mesh.create()




    # create output
    out = fluxy.output(params, mesh)
    out.write()
    out.write(132)

if __name__ == '__main__':
    main()