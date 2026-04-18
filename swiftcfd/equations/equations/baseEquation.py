from abc import ABC, abstractmethod

from petsc4py import PETSc

from swiftcfd.equations.boundaryConditions.boundaryConditions import BoundaryConditions
from swiftcfd.equations.linearAlgebraSolver.linearAlgebraSolver import LinearAlgebraSolver
from swiftcfd.enums import PrimitiveVariables as pv
from swiftcfd.enums import BCType, CornerType

class BaseEquation(ABC):
    def __init__(self, params, mesh, field_manager):
        self.params = params
        self.mesh = mesh
        self.field_manager = field_manager

        self.bc = BoundaryConditions(self.params, self.mesh, self.get_variable_name())

        self.has_first_order_time_derivative = False
        self.has_first_order_space_derivative = False
        self.has_second_order_space_derivative = False
        self.has_source = False

        # specify whether this equation should be linearised (picard iterations will be applied)
        self.requires_linearisation = False

        self.solver = LinearAlgebraSolver(self.params, self.mesh, self.get_variable_name(), self.bc.is_fully_neumann())

        self.field_manager.add_field(self.get_variable_name())

    def pre_solve_task(self, runtime):
        """Anything that needs to be done before the solver is called."""
        pass

    def post_solve_task(self, runtime):
        """Anything that needs to be done after the solver is called."""
        pass

    def solve(self, runtime):
        # ensure matrix and right-hand side vector are zeroed
        self.solver.reset_A()
        self.solver.reset_b()
        
        # set internal matrix coefficients
        if self.has_first_order_time_derivative:
            self.first_order_time_derivative(runtime)
        if self.has_first_order_space_derivative:
            self.first_order_space_derivative(runtime) 
        if self.has_second_order_space_derivative:
            self.second_order_space_derivative(runtime)
        if self.has_source:
            self.source(runtime)

        # assemble matrix
        self.solver.assemble()
        # if self.get_variable_name() == 'p':
        #     self.solver.view()
        #     exit(0)

        # solver linear system of equations after coefficient matrix has been assembled
        self.solver.solve(self.field_manager.fields[self.get_variable_name()])

        # apply under-relaxation
        self.under_relaxation()

    def under_relaxation(self):
        alpha = self.params('solver', 'linearSolver', 'underRelaxation', self.get_variable_name())

        for (block, i, j) in self.mesh.loop_all_cells():
            self.field_manager.fields[self.get_variable_name()][block, i, j] = \
                alpha * self.field_manager.fields[self.get_variable_name()][block, i, j] + \
                (1.0 - alpha) * self.field_manager.fields[self.get_variable_name()].picard_old[block, i, j]

    def first_order_time_derivative(self, runtime):
        """Handle time derivatives of the equation."""
        pass

    def first_order_space_derivative(self, runtime):
        """Handle advection (first-order spatial derivatives)."""
        pass

    def second_order_space_derivative(self, runtime):
        """Handle diffusion (second-order spatial derivatives)."""
        pass

    def source(self, runtime):
        """Handle sources of the equation."""
        pass

    @abstractmethod
    def get_diffusion_coefficients(self):
        pass

    @abstractmethod
    def get_variable_name(self):
        pass