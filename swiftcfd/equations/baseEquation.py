from abc import ABC, abstractmethod

from swiftcfd.equations.boundaryConditions.boundaryConditions import BoundaryConditions
from swiftcfd.equations.boundaryConditions.interfaceConditions import InterfaceConditions
from swiftcfd.equations.boundaryConditions.cornerPoint import CornerPoint
from swiftcfd.equations.linearAlgebraSolver.linearAlgebraSolver import LinearAlgebraSolver

class BaseEquation(ABC):
    def __init__(self, params, mesh, var_name):
        self.params = params
        self.mesh = mesh
        self.var_name = var_name
        self.bc = BoundaryConditions(self.params, self.mesh, self.var_name)
        self.ic = InterfaceConditions(self.mesh, self.bc)
        self.cp = CornerPoint(self.mesh, self.bc)

        self.has_first_order_time_derivative = False
        self.has_first_order_space_derivative = False
        self.has_second_order_space_derivative = False
        self.has_source = False

        self.solver = LinearAlgebraSolver(self.params, self.mesh, self.var_name)

    def update(self, time, field):
        # ensure matrix and right-hand side vector are zeroed
        self.solver.reset_A()
        self.solver.reset_b()
        
        # set internal matrix coefficients
        if self.has_first_order_time_derivative:
            self.first_order_time_derivative(time, field)
        if self.has_first_order_space_derivative:
            self.first_order_space_derivative(time, field)
        if self.has_second_order_space_derivative:
            self.second_order_space_derivative(time, field)
        if self.has_source:
            self.source(time, field)

        # set coefficients based on boundary conditions
        self.bc.apply_boundary_conditions(self.solver, field)

        # set corner points
        self.cp.apply_corner_points(self.solver, field)

        # assemble matrix
        self.solver.assemble()

        # solver linear system of equations after coefficient matrix has been assembled
        self.solver.solve(field)

        # adjust corner points
        self.cp.average_field_at_corner_point(field)

    def first_order_time_derivative(self, time, field):
        """Handle time derivatives of the equation."""
        pass

    def first_order_space_derivative(self, time, field):
        """Handle advection (first-order spatial derivatives)."""
        pass

    def second_order_space_derivative(self, time, field):
        """Handle diffusion (second-order spatial derivatives)."""
        pass

    def source(self, time, field):
        """Handle sources of the equation."""
        pass

    @abstractmethod
    def get_diffusion_coefficients(self):
        pass