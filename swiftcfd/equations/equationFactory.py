from swiftcfd.equations.equations import *

class EquationFactory:
    def __init__(self, params, mesh):
        self.params = params
        self.mesh = mesh

    def create(self, equation_name):
        if equation_name == 'heatDiffusion':
            return heatDiffusion.HeatDiffusion(self.params, self.mesh)
        else:
            print(f'Unknown equation "{equation_name}" selected')
            print('Available equations: heatDiffusion')
            print('Exiting now ...')
            exit(1)