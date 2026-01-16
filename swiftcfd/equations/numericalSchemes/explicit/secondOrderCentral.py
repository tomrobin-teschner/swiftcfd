from swiftcfd.equations.numericalSchemes.numericalSchemesBase import NumericalSchemesBase
from swiftcfd.equations.numericalSchemes.numericalSchemesBase import WRT
from math import pow

class SecondOrderCentral(NumericalSchemesBase):
    def __init__(self, params, mesh, ic, field_manager):
        super().__init__(params, mesh, ic, field_manager)

    def _compute_coefficients(self, direction, time, var_name, multiplier):
        for block_id in range(0, self.mesh.num_blocks):
            dx, dy = self.mesh.get_spacing(block_id)
            self.coefficients.append({
                'b': multiplier
            })
    
    def _compute_interior(self, direction, block_id, solver, var_name):
        dx, dy = self.mesh.get_spacing(block_id)
        for (i, j) in self.mesh.internal_loop_single_block(block_id):
            b_index = self.mesh.map3Dto1D(block_id, i, j)
            ij = (block_id, i, j)
            ip1j = (block_id, i + 1, j)
            im1j = (block_id, i - 1, j)
            ijp1 = (block_id, i, j + 1)
            ijm1 = (block_id, i, j - 1)

            b_value = self.get_right_hand_side_contribution(direction, ij, ip1j, im1j, ijp1, ijm1, var_name)
            solver.add_to_b(b_index, b_value)
    
    def get_right_hand_side_contribution(self, direction, ij, ip1j, im1j, ijp1, ijm1, var_name):
        block1, i1, j1 = ij
        block2, i2, j2 = ip1j
        block3, i3, j3 = im1j
        block4, i4, j4 = ijp1
        block5, i5, j5 = ijm1

        dx, dy = self.mesh.get_spacing(block1)
        multiplier = self.coefficients[block1]['b']

        phi_ij  = self.field_manager.fields[var_name].old[block1, i1, j1]
        if direction == WRT.x:
            phi_ip1 = self.field_manager.fields[var_name].old[block2, i2, j2]
            phi_im1 = self.field_manager.fields[var_name].old[block3, i3, j3]
            grad = multiplier * (phi_ip1 - 2.0 * phi_ij + phi_im1) / pow(dx, 2)
        elif direction == WRT.y:
            phi_jp1 = self.field_manager.fields[var_name].old[block4, i4, j4]
            phi_jm1 = self.field_manager.fields[var_name].old[block5, i5, j5]
            grad = multiplier * (phi_jp1 - 2.0 * phi_ij + phi_jm1) / pow(dy, 2)
        
        return grad