from swiftcfd.parameters import Parameters as parameters
from swiftcfd.mesh import Mesh as mesh
from swiftcfd.output import Output as output
from swiftcfd.field.fieldManager import FieldManager as field_manager
from swiftcfd.time import Time as time
from swiftcfd.equations.equations.heatDiffusion import HeatDiffusion as heat_diffusion
from swiftcfd.log import Log as log
from swiftcfd.performanceStatistics import PerformanceStatistics as performance_statistics
from swiftcfd.equations.equationFactory import EquationFactory as equation_factory

from swiftcfd.solvers import heatDiffusionSolver as heat_diffusion_solver