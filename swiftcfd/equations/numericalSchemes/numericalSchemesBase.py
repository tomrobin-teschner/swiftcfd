from abc import ABC, abstractmethod

from swiftcfd.equations.boundaryConditions.boundaryConditions import BoundaryConditions

class NumericalSchemesBase(ABC):
    def __init__(self, params, mesh, interface_condition):
        self.params = params
        self.mesh = mesh
        self.ic = interface_condition        

    def apply(self, solver, time, field, multiplier = 1.0):
        # get matrix coefficients for all blocks
        coefficients = []
        for block_id in range(0, self.mesh.num_blocks):
            coefficients.append(self.get_coefficients(time, block_id, field, multiplier))

        for block_id in range(0, self.mesh.num_blocks):
            # apply numerical scheme on interior elements
            self._compute_interior(block_id, solver, field, coefficients)

            # apply boundary conditions on internal elements
            self._apply_interface_conditions(block_id, solver, field, coefficients)
    
    @abstractmethod
    def get_coefficients(self, block_id, time, field, multiplier):
        pass
    
    @abstractmethod
    def _compute_interior(self, block_id, solver, field, coefficients):
        pass

    def _apply_interface_conditions(self, block_id, solver, field, coefficients):
        self.ic.apply_interface_conditions(block_id, solver, field, coefficients)
