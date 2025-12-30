from abc import ABC

from fluxy.equations.boundaryConditions.boundaryConditions import BoundaryConditions
from fluxy.equations.linearAlgebraSolver.linearAlgebraSolver import LinearAlgebraSolver

class BaseEquation(ABC):
    def __init__(self, params, mesh, var_name):
        self.params = params
        self.mesh = mesh
        self.var_name = var_name
        self.bc = BoundaryConditions(params, mesh, var_name)

        self.has_time_derivative = False
        self.has_advection = False
        self.has_diffusion = False
        self.has_source = False

        self.solver = LinearAlgebraSolver(params, mesh, var_name)

    def update(self, time, field):
        # ensure matrix and right-hand side vector are zeroed
        self.solver.reset_A()
        self.solver.reset_b()
        
        # set internal matrix coefficients
        if self.has_time_derivative:
            self.time_derivative(time, field)
        if self.has_advection:
            self.advection(time, field)
        if self.has_diffusion:
            self.diffusion(time, field)
        if self.has_source:
            self.source(time, field)

        # set coefficients based on boundary conditions
        self.bc.apply_boundary_conditions(self.solver, field)

        # assemble matrix
        self.solver.assemble()

        # solver linear system of equations after coefficient matrix has been assembled
        self.solver.solve(field)

    def time_derivative(self, time, field):
        """Handle time derivatives of the equation."""
        pass

    def advection(self, time, field):
        """Handle advection (first-order spatial derivatives)."""
        pass

    def diffusion(self, time, field):
        """Handle diffusion (second-order spatial derivatives)."""
        pass

    def source(self, time, field):
        """Handle sources of the equation."""
        pass