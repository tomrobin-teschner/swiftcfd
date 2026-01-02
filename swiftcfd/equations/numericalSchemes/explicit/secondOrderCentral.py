from swiftcfd.equations.numericalSchemes.numericalSchemesBase import NumericalSchemesBase
from math import pow

class SecondOrderCentral(NumericalSchemesBase):
    def __init__(self, params, mesh, bc):
        super().__init__(params, mesh, bc)

    def _compute_coefficients(self, time, field, multiplier):
        for block_id in range(0, self.mesh.num_blocks):
            dx, dy = self.mesh.get_spacing(block_id)
            self.coefficients.append({
                'b': multiplier
            })
    
    def _compute_interior(self, block_id, solver, field):
        dx, dy = self.mesh.get_spacing(block_id)
        for (i, j) in self.mesh.internal_loop_single_block(block_id):
            b_index = self.mesh.map3Dto1D(block_id, i, j)
            ij = (block_id, i, j)
            ip1j = (block_id, i + 1, j)
            im1j = (block_id, i - 1, j)
            ijp1 = (block_id, i, j + 1)
            ijm1 = (block_id, i, j - 1)

            b_value = self.get_right_hand_side_contribution(ij, ip1j, im1j, ijp1, ijm1, field)
            
            # Tij  = field.old[block_id, i, j]
            # Tip1 = field.old[block_id, i + 1, j]
            # Tim1 = field.old[block_id, i - 1, j]
            # Tjp1 = field.old[block_id, i, j + 1]
            # Tjm1 = field.old[block_id, i, j - 1]
            # alpha = self.coefficients[block_id]['b']

            # grad_x = alpha * (Tip1 - 2.0 * Tij + Tim1) / pow(dx, 2)
            # grad_y = alpha * (Tjp1 - 2.0 * Tij + Tjm1) / pow(dy, 2)

            # solver.add_to_b(b_index, grad_x + grad_y)
            solver.add_to_b(b_index, b_value)
    
    def get_right_hand_side_contribution(self, ij, ip1j, im1j, ijp1, ijm1, field):
        block1, i1, j1 = ij
        block2, i2, j2 = ip1j
        block3, i3, j3 = im1j
        block4, i4, j4 = ijp1
        block5, i5, j5 = ijm1

        dx, dy = self.mesh.get_spacing(block1)

        phi_ij  = field.old[block1, i1, j1]
        phi_ip1 = field.old[block2, i2, j2]
        phi_im1 = field.old[block3, i3, j3]
        phi_jp1 = field.old[block4, i4, j4]
        phi_jm1 = field.old[block5, i5, j5]

        alpha = self.coefficients[block1]['b']

        grad_x = alpha * (phi_ip1 - 2.0 * phi_ij + phi_im1) / pow(dx, 2)
        grad_y = alpha * (phi_jp1 - 2.0 * phi_ij + phi_jm1) / pow(dy, 2)

        return grad_x + grad_y
