from swiftcfd.equations.numericalSchemes.numericalSchemesBase import NumericalSchemesBase
from swiftcfd.enums import BCType, WRT
from math import pow

class SecondOrderCentral(NumericalSchemesBase):
    def __init__(self, params, mesh, bc, field_manager):
        super().__init__(params, mesh, bc, field_manager)

    def _compute_coefficients(self, direction, time, var_name, multiplier):
        for block_id in range(0, self.mesh.num_blocks):
            dx, dy = self.mesh.get_spacing(block_id)
            self.coefficients.append({
                'apx': multiplier * (-2.0/pow(dx, 2)),
                'apy': multiplier * (-2.0/pow(dy, 2)),
                'ae': multiplier * ( 1.0/pow(dx, 2)),
                'aw': multiplier * ( 1.0/pow(dx, 2)),
                'an': multiplier * ( 1.0/pow(dy, 2)),
                'as': multiplier * ( 1.0/pow(dy, 2)),
            })
    
    def _compute_interior(self, direction, block_id, solver, var_name):
        for (i, j) in self.mesh.internal_loop_single_block(block_id):

            if direction == WRT.x:
                self.__apply_in_x(solver, block_id, i, j)
                
            elif direction == WRT.y:
                self.__apply_in_y(solver, block_id, i, j)

    def get_right_hand_side_contribution(self, direction, ij, ip1j, im1j, ijp1, ijm1, var_name):
        return 0.0

    def _east_boundary(self, direction, block_id, solver, var_name):
        if self.bc.get_bc_type(block_id, "east") == BCType.neumann:
            if direction == WRT.x:
                for (i, j) in self.mesh.loop_east(block_id, 1):
                    ap_index = self.mesh.map3Dto1D(block_id, i, j)
                    aw_index = self.mesh.map3Dto1D(block_id, i - 1, j)

                    dx, _ = self.mesh.get_spacing(block_id)
                    phi_east = self.bc.get_bc_value(block_id, "east")
                    rhs = -2.0 * phi_east / dx 

                    solver.add_to_A(ap_index, ap_index, self.coefficients[block_id]['apx'])
                    solver.add_to_A(ap_index, aw_index, 2.0 * self.coefficients[block_id]['aw'])

                    solver.add_to_b(ap_index, rhs)
            elif direction == WRT.y:
                for (i, j) in self.mesh.loop_east(block_id, 1):
                    self.__apply_in_y(solver, block_id, i, j)
        # else:
        #     raise NotImplementedError

    def _west_boundary(self, direction, block_id, solver, var_name):
        if self.bc.get_bc_type(block_id, "west") == BCType.neumann:
            if direction == WRT.x:
                for (i, j) in self.mesh.loop_west(block_id, 1):
                    ap_index = self.mesh.map3Dto1D(block_id, i, j)
                    ae_index = self.mesh.map3Dto1D(block_id, i + 1, j)

                    dx, _ = self.mesh.get_spacing(block_id)
                    phi_west = self.bc.get_bc_value(block_id, "west")
                    rhs = 2.0 * phi_west / dx 

                    solver.add_to_A(ap_index, ap_index, self.coefficients[block_id]['apx'])
                    solver.add_to_A(ap_index, ae_index, 2.0 * self.coefficients[block_id]['ae'])

                    solver.add_to_b(ap_index, rhs)
            elif direction == WRT.y:
                for (i, j) in self.mesh.loop_west(block_id, 1):
                    self.__apply_in_y(solver, block_id, i, j)
        # else:
        #     raise NotImplementedError

    def _north_boundary(self, direction, block_id, solver, var_name):
        if self.bc.get_bc_type(block_id, "north") == BCType.neumann:
            if direction == WRT.x:
                for (i, j) in self.mesh.loop_north(block_id, 1):
                    self.__apply_in_x(solver, block_id, i, j)

            elif direction == WRT.y:
                for (i, j) in self.mesh.loop_north(block_id, 1):
                    ap_index = self.mesh.map3Dto1D(block_id, i, j)
                    as_index = self.mesh.map3Dto1D(block_id, i, j - 1)

                    _, dy = self.mesh.get_spacing(block_id)
                    phi_north = self.bc.get_bc_value(block_id, "north")
                    rhs = -2.0 * phi_north / dy 

                    solver.add_to_A(ap_index, ap_index, self.coefficients[block_id]['apy'])
                    solver.add_to_A(ap_index, as_index, 2.0 * self.coefficients[block_id]['as'])

                    solver.add_to_b(ap_index, rhs)
        # else:
        #     raise NotImplementedError

    def _south_boundary(self, direction, block_id, solver, var_name):
        if self.bc.get_bc_type(block_id, "south") == BCType.neumann:
            if direction == WRT.x:
                for (i, j) in self.mesh.loop_south(block_id, 1):
                    self.__apply_in_x(solver, block_id, i, j)
                
            elif direction == WRT.y:
                for (i, j) in self.mesh.loop_south(block_id, 1):
                    ap_index = self.mesh.map3Dto1D(block_id, i, j)
                    an_index = self.mesh.map3Dto1D(block_id, i, j + 1)

                    _, dy = self.mesh.get_spacing(block_id)
                    phi_south = self.bc.get_bc_value(block_id, "south")
                    rhs = 2.0 * phi_south / dy 

                    solver.add_to_A(ap_index, ap_index, self.coefficients[block_id]['apy'])
                    solver.add_to_A(ap_index, an_index, 2.0 * self.coefficients[block_id]['an'])

                    solver.add_to_b(ap_index, rhs)
                
        # else:
        #     raise NotImplementedError
    
    def __apply_in_x(self, solver, block_id, i, j):
        ap_index = self.mesh.map3Dto1D(block_id, i, j)
        ae_index = self.mesh.map3Dto1D(block_id, i + 1, j)
        aw_index = self.mesh.map3Dto1D(block_id, i - 1, j)

        solver.add_to_A(ap_index, ap_index, self.coefficients[block_id]['apx'])
        solver.add_to_A(ap_index, ae_index, self.coefficients[block_id]['ae'])
        solver.add_to_A(ap_index, aw_index, self.coefficients[block_id]['aw'])

    def __apply_in_y(self, solver, block_id, i, j):
        ap_index = self.mesh.map3Dto1D(block_id, i, j)
        an_index = self.mesh.map3Dto1D(block_id, i, j + 1)
        as_index = self.mesh.map3Dto1D(block_id, i, j - 1)

        solver.add_to_A(ap_index, ap_index, self.coefficients[block_id]['apy'])
        solver.add_to_A(ap_index, an_index, self.coefficients[block_id]['an'])
        solver.add_to_A(ap_index, as_index, self.coefficients[block_id]['as'])