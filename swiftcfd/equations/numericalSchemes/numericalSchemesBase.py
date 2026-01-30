from abc import ABC, abstractmethod
from enum import Enum, auto

from swiftcfd.equations.boundaryConditions.boundaryConditions import BoundaryConditions

class NumericalSchemesBase(ABC):
    def __init__(self, params, mesh, boundary_conditions, field_manager):
        self.params = params
        self.mesh = mesh
        self.bc = boundary_conditions
        self.field_manager = field_manager
        self.coefficients = []

    def apply(self, direction, solver, time, var_name, multiplier = 1.0):
        # compute latest coefficients
        self.coefficients = []
        self._compute_coefficients(direction, time, var_name, multiplier)

        # get matrix coefficients for all blocks
        for block_id in range(0, self.mesh.num_blocks):
            # apply numerical scheme on interior elements
            self._compute_interior(direction, block_id, solver, var_name)

            # set dirichlet boundary conditions

            # apply numerical scheme on boundary elements
            self._east_boundary(direction, block_id, solver, var_name)
            self._west_boundary(direction, block_id, solver, var_name)
            self._north_boundary(direction, block_id, solver, var_name)
            self._south_boundary(direction, block_id, solver, var_name)

            # # TODO: remove this! ... apply boundary conditions on internal elements
            # self._apply_interface_conditions(block_id, solver, var_name)

    @abstractmethod
    def _east_boundary(self, direction, block_id, solver, var_name):
        pass

    @abstractmethod
    def _west_boundary(self, direction, block_id, solver, var_name):
        pass

    @abstractmethod
    def _north_boundary(self, direction, block_id, solver, var_name):
        pass

    @abstractmethod
    def _south_boundary(self, direction, block_id, solver, var_name):
        pass

    @abstractmethod
    def _compute_coefficients(self, direction, block_id, time, var_name, multiplier):
        pass
    
    @abstractmethod
    def _compute_interior(self, direction, block_id, solver, var_name):
        pass

    
    
    @abstractmethod
    def get_right_hand_side_contribution(self, direction, ij, ip1j, im1j, ijp1, ijm1, var_name):
        pass

    # def _apply_interface_conditions(self, block_id, solver, var_name):
    #     self.ic.apply_interface_conditions(block_id, solver, var_name, self)
