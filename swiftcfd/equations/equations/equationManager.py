from swiftcfd.equations.equations.equationFactory import EquationFactory as equation_factory

from petsc4py import init as petsc_init

class EquationManager:
    def __init__(self, params, mesh):
        # initialise petsc for usage, required by the linear solver within each equation
        petsc_init()

        self.params = params
        self.mesh = mesh
        self.equations, self.field_manager = equation_factory(params, mesh).create()

        # create a list of linear and non-linear equations
        self.non_linear_equations = [eqn for eqn in self.equations if eqn.requires_linearisation]
        self.linear_equations = [eqn for eqn in self.equations if not eqn.requires_linearisation]

    def solve_non_linear_equations(self, runtime, stats):
        self.solve(runtime, stats, self.non_linear_equations)

    def solve_linear_equations(self, runtime, stats):
        self.solve(runtime, stats, self.linear_equations)

    def solve(self, runtime, stats, equations):
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
