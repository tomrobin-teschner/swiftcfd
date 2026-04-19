from swiftcfd.equations.numericalSchemes.implicit.firstOrderUpwind import FirstOrderUpwind
from swiftcfd.enums import WRT, BCType
from swiftcfd.enums import PrimitiveVariables as pv

class SecondOrderUpwind(FirstOrderUpwind):
    def __init__(self, params, mesh, bc, field_manager):
        super().__init__(params, mesh, bc, field_manager)

        self.inv_2dx = []
        self.inv_2dy = []
        for block_id in range(self.mesh.num_blocks):
            dx, dy = self.mesh.get_spacing(block_id)
            self.inv_2dx.append(1.0 / (2.0 * dx))
            self.inv_2dy.append(1.0 / (2.0 * dy))
    
    def _compute_interior(self, direction, block_id, solver, var_name):
        # loop over internal cells, with an offset of 1 cell from the boundary
        # and apply smaller first-order upwind stencil to avoid overshooting boundaries
        for (i, j) in self.mesh.loop_cells_with_offset_from_boundary(block_id, 1):
            if direction == WRT.x:
                self.upwind_wrt_x(block_id, i, j, solver)

            elif direction == WRT.y:
                self.upwind_wrt_y(block_id, i, j, solver)
        
        # loop over internal cells, with an offset of 2 from the x and y boundary
        # so that second-order stencil does not go over boundary
        for (i, j) in self.mesh.loop_cells_with_offset(block_id, 2, 2):
            if direction == WRT.x:
                self.upwind_wrt_x_2nd_order(block_id, i, j, solver)

            elif direction == WRT.y:
                self.upwind_wrt_y_2nd_order(block_id, i, j, solver)
    
    def upwind_wrt_x_2nd_order(self, block_id, i, j, solver):
        self.positive_in_x_2nd_order(block_id, i, j, solver)
        self.negative_in_x_2nd_order(block_id, i, j, solver)

    def upwind_wrt_y_2nd_order(self, block_id, i, j, solver):
        self.positive_in_y_2nd_order(block_id, i, j, solver)
        self.negative_in_y_2nd_order(block_id, i, j, solver)
    
    def positive_in_x_2nd_order(self, block_id, i, j, solver):
        max_u = max(self.field_manager.fields[pv.velocity_x.name()].picard_old[block_id, i, j], 0.0)

        ap_index  = self.mesh.map3Dto1D(block_id, i, j)
        aw_index  = self.mesh.map3Dto1D(block_id, i - 1, j)
        aww_index = self.mesh.map3Dto1D(block_id, i - 2, j)
        
        ap_value  =  3.0 * max_u * self.inv_2dx[block_id]
        aw_value  = -4.0 * max_u * self.inv_2dx[block_id]
        aww_value =  1.0 * max_u * self.inv_2dx[block_id]

        solver.add_to_A(ap_index, ap_index,  ap_value )
        solver.add_to_A(ap_index, aw_index,  aw_value )
        solver.add_to_A(ap_index, aww_index, aww_value)

    def negative_in_x_2nd_order(self, block_id, i, j, solver):
        min_u = min(self.field_manager.fields[pv.velocity_x.name()].picard_old[block_id, i, j], 0.0)

        ap_index  = self.mesh.map3Dto1D(block_id, i, j)
        ae_index  = self.mesh.map3Dto1D(block_id, i + 1, j)
        aee_index = self.mesh.map3Dto1D(block_id, i + 2, j)

        ap_value  = -3.0 * min_u * self.inv_2dx[block_id]
        ae_value  =  4.0 * min_u * self.inv_2dx[block_id]
        aee_value = -1.0 * min_u * self.inv_2dx[block_id]

        solver.add_to_A(ap_index, ap_index,  ap_value )
        solver.add_to_A(ap_index, ae_index,  ae_value )
        solver.add_to_A(ap_index, aee_index, aee_value)

    def positive_in_y_2nd_order(self, block_id, i, j, solver):
        max_v = max(self.field_manager.fields[pv.velocity_y.name()].picard_old[block_id, i, j], 0.0)

        ap_index  = self.mesh.map3Dto1D(block_id, i, j)
        as_index  = self.mesh.map3Dto1D(block_id, i, j - 1)
        ass_index = self.mesh.map3Dto1D(block_id, i, j - 2)

        ap_value  =  3.0 * max_v * self.inv_2dy[block_id]
        as_value  = -4.0 * max_v * self.inv_2dy[block_id]
        ass_value =  1.0 * max_v * self.inv_2dy[block_id]

        solver.add_to_A(ap_index, ap_index,  ap_value )
        solver.add_to_A(ap_index, as_index,  as_value )
        solver.add_to_A(ap_index, ass_index, ass_value)
    
    def negative_in_y_2nd_order(self, block_id, i, j, solver):
        min_v = min(self.field_manager.fields[pv.velocity_y.name()].picard_old[block_id, i, j], 0.0)

        ap_index  = self.mesh.map3Dto1D(block_id, i, j)
        an_index  = self.mesh.map3Dto1D(block_id, i, j + 1)
        ann_index = self.mesh.map3Dto1D(block_id, i, j + 2)

        ap_value  = -3.0 * min_v * self.inv_2dy[block_id]
        an_value  =  4.0 * min_v * self.inv_2dy[block_id]
        ann_value = -1.0 * min_v * self.inv_2dy[block_id]

        solver.add_to_A(ap_index, ap_index,  ap_value )
        solver.add_to_A(ap_index, an_index,  an_value )
        solver.add_to_A(ap_index, ann_index, ann_value)
