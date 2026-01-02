from abc import ABC, abstractmethod

from swiftcfd.equations.boundaryConditions.boundaryConditions import BoundaryConditions

class NumericalSchemesBase(ABC):
    def __init__(self, params, mesh, interface_condition):
        self.params = params
        self.mesh = mesh
        self.ic = interface_condition
        self.coefficients = []

    def apply(self, solver, time, field, multiplier = 1.0):
        # compute latest coefficients
        self.coefficients = []
        self._compute_coefficients(time, field, multiplier)

        # get matrix coefficients for all blocks
        for block_id in range(0, self.mesh.num_blocks):
            # apply numerical scheme on interior elements
            self._compute_interior(block_id, solver, field)

            # apply boundary conditions on internal elements
            self._apply_interface_conditions(block_id, solver, field)
    
    @abstractmethod
    def _compute_coefficients(self, block_id, time, field, multiplier):
        pass
    
    @abstractmethod
    def _compute_interior(self, block_id, solver, field):
        pass

    def _apply_interface_conditions(self, block_id, solver, field):
        self.ic.apply_interface_conditions(block_id, solver, field, self.coefficients)
