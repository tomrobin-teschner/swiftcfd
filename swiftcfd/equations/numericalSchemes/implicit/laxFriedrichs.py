from swiftcfd.equations.numericalSchemes.numericalSchemesBase import NumericalSchemesBase
from swiftcfd.enums import WRT
from math import pow

class LaxFriedrichs(NumericalSchemesBase):
    def __init__(self, params, mesh, ic, field_manager):
        super().__init__(params, mesh, ic, field_manager)

    def _compute_coefficients(self, direction, time, var_name, multiplier):
        for block_id in range(0, self.mesh.num_blocks):
            dx, dy = self.mesh.get_spacing(block_id)
            dt = time.dt
            if direction == WRT.x:
                self.coefficients.append({
                    'ap': 0.5,
                    'ae': -0.25,
                    'aw': -0.25,
                    'dt': dt,
                })
            elif direction == WRT.y:
                self.coefficients.append({
                    'ap': 0.5,
                    'an': -0.25,
                    'as': -0.25,
                    'dt': dt,
                })
            else:
                raise NotImplementedError
    
    def _compute_interior(self, direction, block_id, solver, var_name):
        dx, dy = self.mesh.get_spacing(block_id)
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
        block1, i1, j1 = ij
        block2, i2, j2 = ip1j
        block3, i3, j3 = im1j
        block4, i4, j4 = ijp1
        block5, i5, j5 = ijm1

        dx, dy = self.mesh.get_spacing(block1)
        dt = self.coefficients[block1]['dt']

        phi_ij  = self.field_manager.fields[var_name].old[block1, i1, j1]
        if direction == WRT.x:
            phi_ip1 = self.field_manager.fields[var_name].old[block2, i2, j2]
            phi_im1 = self.field_manager.fields[var_name].old[block3, i3, j3]
            RHS = phi_ij - (dt / (2.0 * dx)) * (phi_ip1 - phi_im1)
        elif direction == WRT.y:
            phi_jp1 = self.field_manager.fields[var_name].old[block4, i4, j4]
            phi_jm1 = self.field_manager.fields[var_name].old[block5, i5, j5]
            RHS = phi_ij - (dt / (2.0 * dy)) * (phi_jp1 - phi_jm1)
        
        return RHS