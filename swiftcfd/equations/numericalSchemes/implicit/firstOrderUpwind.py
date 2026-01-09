from swiftcfd.equations.numericalSchemes.numericalSchemesBase import NumericalSchemesBase
from swiftcfd.equations.numericalSchemes.numericalSchemesBase import WRT

class FirstOrderUpwind(NumericalSchemesBase):
    def __init__(self, params, mesh, bc):
        super().__init__(params, mesh, bc)

    def _compute_coefficients(self, direction, time, field, multiplier):
        pass
    
    def _compute_interior(self, direction, block_id, solver, field):
        dx, dy = self.mesh.get_spacing(block_id)
        inv_dx, inv_dy = 1.0 / dx, 1.0 / dy

        for (i, j) in self.mesh.internal_loop_single_block(block_id):
            max_phi = max(field.old[block_id, i, j], 0.0)
            min_phi = min(field.old[block_id, i, j], 0.0)
            
            ap_index = self.mesh.map3Dto1D(block_id, i, j)
            ap_value = max_phi * inv_dx - min_phi * inv_dy
            solver.add_to_A(ap_index, ap_index, ap_value)

            if direction == WRT.x:
                ae_index = self.mesh.map3Dto1D(block_id, i + 1, j)
                aw_index = self.mesh.map3Dto1D(block_id, i - 1, j)

                ae_value =  inv_dx * min_phi
                aw_value = -inv_dx * max_phi

                solver.add_to_A(ap_index, ae_index, ae_value)
                solver.add_to_A(ap_index, aw_index, aw_value)

            elif direction == WRT.y:
                an_index = self.mesh.map3Dto1D(block_id, i, j + 1)
                as_index = self.mesh.map3Dto1D(block_id, i, j - 1)

                an_value =  inv_dy * min_phi
                as_value = -inv_dy * max_phi

                solver.add_to_A(ap_index, an_index, an_value)
                solver.add_to_A(ap_index, as_index, as_value)

    def get_right_hand_side_contribution(self, direction, ij, ip1j, im1j, ijp1, ijm1, field):
        return 0.0