from swiftcfd.equations.numericalSchemes.numericalSchemesBase import NumericalSchemesBase
from swiftcfd.enums import WRT, BCType
from swiftcfd.enums import PrimitiveVariables as pv

class FirstOrderUpwind(NumericalSchemesBase):
    def __init__(self, params, mesh, bc, field_manager):
        super().__init__(params, mesh, bc, field_manager)

    def _compute_coefficients(self, direction, time, var_name, multiplier):
        pass
    
    def _compute_interior(self, direction, block_id, solver, var_name):
        dx, dy = self.mesh.get_spacing(block_id)
        inv_dx, inv_dy = 1.0 / dx, 1.0 / dy

        for (i, j) in self.mesh.internal_loop_single_block(block_id):
            
            if direction == WRT.x:
                max_u = max(self.field_manager.fields[pv.velocity_x.name()].picard_old[block_id, i, j], 0.0)
                min_u = min(self.field_manager.fields[pv.velocity_x.name()].picard_old[block_id, i, j], 0.0)

                ap_index = self.mesh.map3Dto1D(block_id, i, j)
                ae_index = self.mesh.map3Dto1D(block_id, i + 1, j)
                aw_index = self.mesh.map3Dto1D(block_id, i - 1, j)

                ap_value = (max_u - min_u) * inv_dx
                ae_value =  min_u * inv_dx
                aw_value = -1.0 * max_u * inv_dx

                solver.add_to_A(ap_index, ap_index, ap_value)
                solver.add_to_A(ap_index, ae_index, ae_value)
                solver.add_to_A(ap_index, aw_index, aw_value)

            elif direction == WRT.y:
                max_v = max(self.field_manager.fields[pv.velocity_y.name()].picard_old[block_id, i, j], 0.0)
                min_v = min(self.field_manager.fields[pv.velocity_y.name()].picard_old[block_id, i, j], 0.0)

                ap_index = self.mesh.map3Dto1D(block_id, i, j)
                an_index = self.mesh.map3Dto1D(block_id, i, j + 1)
                as_index = self.mesh.map3Dto1D(block_id, i, j - 1)

                ap_value = (max_v - min_v) * inv_dy
                an_value =  min_v * inv_dy
                as_value = -1.0 * max_v * inv_dy

                solver.add_to_A(ap_index, ap_index, ap_value)
                solver.add_to_A(ap_index, an_index, an_value)
                solver.add_to_A(ap_index, as_index, as_value)

    def get_right_hand_side_contribution(self, direction, ij, ip1j, im1j, ijp1, ijm1, var_name):
        return 0.0

    def _east_boundary(self, direction, block_id, solver, var_name):
        dx, dy = self.mesh.get_spacing(block_id)
        inv_dx, inv_dy = 1.0 / dx, 1.0 / dy

        if self.bc.get_bc_type(block_id, "east") == BCType.neumann:
            phi_east = self.bc.get_bc_value(block_id, "east")
            for (i, j) in self.mesh.loop_east(block_id, 1):
                if direction == WRT.x:
                    max_u = max(self.field_manager.fields[pv.velocity_x.name()].picard_old[block_id, i, j], 0.0)
                    min_u = min(self.field_manager.fields[pv.velocity_x.name()].picard_old[block_id, i, j], 0.0)

                    ap_index = self.mesh.map3Dto1D(block_id, i, j)
                    aw_index = self.mesh.map3Dto1D(block_id, i - 1, j)

                    ap_value = (max_u - min_u) * inv_dx
                    aw_value = (min_u - max_u) * inv_dx

                    rhs = -2.0 * min_u * phi_east

                    solver.add_to_A(ap_index, ap_index, ap_value)
                    solver.add_to_A(ap_index, aw_index, aw_value)
                    solver.add_to_b(ap_index, rhs)

                elif direction == WRT.y:
                    max_v = max(self.field_manager.fields[pv.velocity_y.name()].picard_old[block_id, i, j], 0.0)
                    min_v = min(self.field_manager.fields[pv.velocity_y.name()].picard_old[block_id, i, j], 0.0)

                    ap_index = self.mesh.map3Dto1D(block_id, i, j)
                    an_index = self.mesh.map3Dto1D(block_id, i, j + 1)
                    as_index = self.mesh.map3Dto1D(block_id, i, j - 1)

                    ap_value = (max_v - min_v) * inv_dy
                    an_value =  min_v * inv_dy
                    as_value = -1.0 * max_v * inv_dy

                    solver.add_to_A(ap_index, ap_index, ap_value)
                    solver.add_to_A(ap_index, an_index, an_value)
                    solver.add_to_A(ap_index, as_index, as_value)
        # else:
        #     raise NotImplementedError

    def _west_boundary(self, direction, block_id, solver, var_name):
        dx, dy = self.mesh.get_spacing(block_id)
        inv_dx, inv_dy = 1.0 / dx, 1.0 / dy

        if self.bc.get_bc_type(block_id, "west") == BCType.neumann:
            phi_west = self.bc.get_bc_value(block_id, "west")
            for (i, j) in self.mesh.loop_west(block_id, 1):
                if direction == WRT.x:
                    max_u = max(self.field_manager.fields[pv.velocity_x.name()].picard_old[block_id, i, j], 0.0)
                    min_u = min(self.field_manager.fields[pv.velocity_x.name()].picard_old[block_id, i, j], 0.0)

                    ap_index = self.mesh.map3Dto1D(block_id, i, j)
                    ae_index = self.mesh.map3Dto1D(block_id, i + 1, j)

                    ap_value = (max_u - min_u) * inv_dx
                    ae_value = (min_u - max_u) * inv_dx

                    rhs = 2.0 * max_u * phi_west

                    solver.add_to_A(ap_index, ap_index, ap_value)
                    solver.add_to_A(ap_index, ae_index, ae_value)
                    solver.add_to_b(ap_index, rhs)

                elif direction == WRT.y:
                    max_v = max(self.field_manager.fields[pv.velocity_y.name()].picard_old[block_id, i, j], 0.0)
                    min_v = min(self.field_manager.fields[pv.velocity_y.name()].picard_old[block_id, i, j], 0.0)

                    ap_index = self.mesh.map3Dto1D(block_id, i, j)
                    an_index = self.mesh.map3Dto1D(block_id, i, j + 1)
                    as_index = self.mesh.map3Dto1D(block_id, i, j - 1)

                    ap_value = (max_v - min_v) * inv_dy
                    an_value =  min_v * inv_dy
                    as_value = -1.0 * max_v * inv_dy

                    solver.add_to_A(ap_index, ap_index, ap_value)
                    solver.add_to_A(ap_index, an_index, an_value)
                    solver.add_to_A(ap_index, as_index, as_value)
        # else:
        #     raise NotImplementedError

    def _north_boundary(self, direction, block_id, solver, var_name):
        dx, dy = self.mesh.get_spacing(block_id)
        inv_dx, inv_dy = 1.0 / dx, 1.0 / dy

        if self.bc.get_bc_type(block_id, "north") == BCType.neumann:
            phi_north = self.bc.get_bc_value(block_id, "north")
            for (i, j) in self.mesh.loop_north(block_id, 1):
                if direction == WRT.x:
                    max_u = max(self.field_manager.fields[pv.velocity_x.name()].picard_old[block_id, i, j], 0.0)
                    min_u = min(self.field_manager.fields[pv.velocity_x.name()].picard_old[block_id, i, j], 0.0)

                    ap_index = self.mesh.map3Dto1D(block_id, i, j)
                    ae_index = self.mesh.map3Dto1D(block_id, i + 1, j)
                    aw_index = self.mesh.map3Dto1D(block_id, i - 1, j)

                    ap_value = (max_u - min_u) * inv_dx
                    ae_value =  min_u * inv_dx
                    aw_value = -1.0 * max_u * inv_dx

                    solver.add_to_A(ap_index, ap_index, ap_value)
                    solver.add_to_A(ap_index, ae_index, ae_value)
                    solver.add_to_A(ap_index, aw_index, aw_value)

                elif direction == WRT.y:
                    max_v = max(self.field_manager.fields[pv.velocity_y.name()].picard_old[block_id, i, j], 0.0)
                    min_v = min(self.field_manager.fields[pv.velocity_y.name()].picard_old[block_id, i, j], 0.0)

                    ap_index = self.mesh.map3Dto1D(block_id, i, j)
                    as_index = self.mesh.map3Dto1D(block_id, i, j - 1)

                    ap_value = (max_v - min_v) * inv_dy
                    as_value = (min_v - max_v) * inv_dy

                    rhs = -2.0 * min_v * phi_north

                    solver.add_to_A(ap_index, ap_index, ap_value)
                    solver.add_to_A(ap_index, as_index, as_value)
                    solver.add_to_b(ap_index, rhs)
            # else:
            #     raise NotImplementedError

    def _south_boundary(self, direction, block_id, solver, var_name):
        dx, dy = self.mesh.get_spacing(block_id)
        inv_dx, inv_dy = 1.0 / dx, 1.0 / dy

        if self.bc.get_bc_type(block_id, "south") == BCType.neumann:
            phi_south = self.bc.get_bc_value(block_id, "south")
            for (i, j) in self.mesh.loop_south(block_id, 1):
                if direction == WRT.x:
                    max_u = max(self.field_manager.fields[pv.velocity_x.name()].picard_old[block_id, i, j], 0.0)
                    min_u = min(self.field_manager.fields[pv.velocity_x.name()].picard_old[block_id, i, j], 0.0)

                    ap_index = self.mesh.map3Dto1D(block_id, i, j)
                    ae_index = self.mesh.map3Dto1D(block_id, i + 1, j)
                    aw_index = self.mesh.map3Dto1D(block_id, i - 1, j)

                    ap_value = (max_u - min_u) * inv_dx
                    ae_value =  min_u * inv_dx
                    aw_value = -1.0 * max_u * inv_dx

                    solver.add_to_A(ap_index, ap_index, ap_value)
                    solver.add_to_A(ap_index, ae_index, ae_value)
                    solver.add_to_A(ap_index, aw_index, aw_value)

                elif direction == WRT.y:
                    max_v = max(self.field_manager.fields[pv.velocity_y.name()].picard_old[block_id, i, j], 0.0)
                    min_v = min(self.field_manager.fields[pv.velocity_y.name()].picard_old[block_id, i, j], 0.0)

                    ap_index = self.mesh.map3Dto1D(block_id, i, j)
                    an_index = self.mesh.map3Dto1D(block_id, i, j + 1)

                    ap_value = (max_v - min_v) * inv_dy
                    an_value = (min_v - max_v) * inv_dy

                    rhs = 2.0 * max_v * phi_south

                    solver.add_to_A(ap_index, ap_index, ap_value)
                    solver.add_to_A(ap_index, an_index, an_value)
                    solver.add_to_b(ap_index, rhs)
            # else:
            #     raise NotImplementedError