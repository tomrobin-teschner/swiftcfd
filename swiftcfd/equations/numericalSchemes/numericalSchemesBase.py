from abc import ABC, abstractmethod
from enum import Enum, auto

from swiftcfd.equations.boundaryConditions.boundaryConditions import BoundaryConditions

# WRT = with respect to, indicates which direction the numerical scheme is applied to
class WRT(Enum):
    t = auto()
    x = auto()
    y = auto()

class NumericalSchemesBase(ABC):
    def __init__(self, params, mesh, interface_condition):
        self.params = params
        self.mesh = mesh
        self.ic = interface_condition
        self.coefficients = []

    def apply(self, direction, solver, time, field, multiplier = 1.0):
        # compute latest coefficients
        self.coefficients = []
        self._compute_coefficients(direction, time, field, multiplier)

        # get matrix coefficients for all blocks
        for block_id in range(0, self.mesh.num_blocks):
            # apply numerical scheme on interior elements
            self._compute_interior(direction, block_id, solver, field)

            # apply boundary conditions on internal elements
            self._apply_interface_conditions(block_id, solver, field)
    
    @abstractmethod
    def _compute_coefficients(self, direction, block_id, time, field, multiplier):
        pass
    
    @abstractmethod
    def _compute_interior(self, direction, block_id, solver, field):
        pass
    
    @abstractmethod
    def get_right_hand_side_contribution(self, direction, ij, ip1j, im1j, ijp1, ijm1, field):
        pass

    def _apply_interface_conditions(self, block_id, solver, field):
        self.ic.apply_interface_conditions(block_id, solver, field, self)
