from swiftcfd.equations.numericalSchemes.numericalSchemesBase import NumericalSchemesBase
from swiftcfd.enums import WRT
from math import pow

class SecondOrderCentral(NumericalSchemesBase):
    def __init__(self, params, mesh, ic, field_manager):
        super().__init__(params, mesh, ic, field_manager)

    def _compute_coefficients(self, direction, time, var_name, multiplier):
        for block_id in range(0, self.mesh.num_blocks):
            dx, dy = self.mesh.get_spacing(block_id)
            if direction == WRT.x:
                self.coefficients.append({
                    'ap': multiplier * (-2.0/pow(dx, 2)),
                    'ae': multiplier * ( 1.0/pow(dx, 2)),
                    'aw': multiplier * ( 1.0/pow(dx, 2)),
                })
            elif direction == WRT.y:
                self.coefficients.append({
                    'ap': multiplier * (-2.0/pow(dy, 2)),
                    'an': multiplier * ( 1.0/pow(dy, 2)),
                    'as': multiplier * ( 1.0/pow(dy, 2)),
                })
            else:
                raise NotImplementedError
    
    def _compute_interior(self, direction, block_id, solver, var_name):
        for (i, j) in self.mesh.internal_loop_single_block(block_id):
            ap_index = self.mesh.map3Dto1D(block_id, i, j)
            solver.add_to_A(ap_index, ap_index, self.coefficients[block_id]['ap'])

            if direction == WRT.x:
                ae_index = self.mesh.map3Dto1D(block_id, i + 1, j)
                aw_index = self.mesh.map3Dto1D(block_id, i - 1, j)
                solver.add_to_A(ap_index, ae_index, self.coefficients[block_id]['ae'])
                solver.add_to_A(ap_index, aw_index, self.coefficients[block_id]['aw'])
            elif direction == WRT.y:
                an_index = self.mesh.map3Dto1D(block_id, i, j + 1)
                as_index = self.mesh.map3Dto1D(block_id, i, j - 1)
                solver.add_to_A(ap_index, an_index, self.coefficients[block_id]['an'])
                solver.add_to_A(ap_index, as_index, self.coefficients[block_id]['as'])
    
    def get_right_hand_side_contribution(self, direction, ij, ip1j, im1j, ijp1, ijm1, var_name):
        return 0.0