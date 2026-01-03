import swiftcfd
   
if __name__ == '__main__':
    parameter_file = 'input/heatedCavity.toml'
    swiftcfd.heat_diffusion_solver.run(parameter_file)