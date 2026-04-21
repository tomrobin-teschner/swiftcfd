from swiftcfd.field.fieldManager import FieldManager

from swiftcfd.equations.equations.heatDiffusion.heatDiffusion import HeatDiffusion

from swiftcfd.equations.equations.fspp.xMomentum import xMomentum as xMomentum
from swiftcfd.equations.equations.fspp.yMomentum import yMomentum as yMomentum
from swiftcfd.equations.equations.fspp.pressure import Pressure as fspp_pressure
from swiftcfd.equations.equations.fsvp.pressure import Pressure as fsvp_pressure


class EquationFactory:
    def __init__(self, params, mesh):
        self.params = params
        self.mesh = mesh

    def create(self):
        # get the name of the solver to instantiate
        solver_name = self.params('solver', 'equation', 'solver')
        
        # create the list of equations to solve
        equations = []
        
        # create the field manager instance
        field_manager = FieldManager(self.mesh)

        if solver_name == 'heatDiffusion':
            equations.append(HeatDiffusion(self.params, self.mesh, field_manager))
        elif solver_name == 'pressureProjection':
            equations.append(xMomentum(self.params, self.mesh, field_manager))
            equations.append(yMomentum(self.params, self.mesh, field_manager))
            equations.append(fspp_pressure(self.params, self.mesh, field_manager))
        elif solver_name == 'fsvp':
            equations.append(xMomentum(self.params, self.mesh, field_manager))
            equations.append(yMomentum(self.params, self.mesh, field_manager))
            equations.append(fsvp_pressure(self.params, self.mesh, field_manager))
        else:
            print(f'Unknown solver "{solver_name}" selected')
            print('Available equations: heatDiffusion, pressureProjection')
            print('Exiting now ...')
            exit(1)

        return equations, field_manager