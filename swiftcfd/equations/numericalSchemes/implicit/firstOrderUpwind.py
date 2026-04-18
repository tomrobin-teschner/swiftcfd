from swiftcfd.equations.numericalSchemes.numericalSchemesBase import NumericalSchemesBase
from swiftcfd.enums import WRT, BCType, CornerType
from swiftcfd.enums import PrimitiveVariables as pv

class FirstOrderUpwind(NumericalSchemesBase):
    def __init__(self, params, mesh, bc, field_manager):
        super().__init__(params, mesh, bc, field_manager)

        self.inv_dx = []
        self.inv_dy = []
        for block_id in range(self.mesh.num_blocks):
            dx, dy = self.mesh.get_spacing(block_id)
            self.inv_dx.append(1.0 / dx)
            self.inv_dy.append(1.0 / dy)

    def _compute_coefficients(self, direction, time, var_name, multiplier):
        pass
    
    def _compute_interior(self, direction, block_id, solver, var_name):
        for (i, j) in self.mesh.loop_internal_cells(block_id):
            if direction == WRT.x:
                self.upwind_wrt_x(block_id, i, j, solver)

            elif direction == WRT.y:
                self.upwind_wrt_y(block_id, i, j, solver)

    def _east_boundary(self, direction, block_id, solver, var_name):
        if direction == WRT.x:
            for (i, j) in self.mesh.loop_east_bc(block_id):
                self.__apply_east_boundary(block_id, i, j, solver)

        elif direction == WRT.y:
            for (i, j) in self.mesh.loop_east_bc(block_id, 1):
                self.upwind_wrt_y(block_id, i, j, solver)
            
            # corner cells
            self.__apply_north_boundary(block_id, self.mesh.num_cells_x[block_id] - 1, self.mesh.num_cells_y[block_id] - 1, solver)        
            self.__apply_south_boundary(block_id, self.mesh.num_cells_x[block_id] - 1, 0, solver)        

    def _west_boundary(self, direction, block_id, solver, var_name):
        if direction == WRT.x:
            for (i, j) in self.mesh.loop_west_bc(block_id):
                self.__apply_west_boundary(block_id, i, j, solver)

        elif direction == WRT.y:
            for (i, j) in self.mesh.loop_west_bc(block_id, 1):
                self.upwind_wrt_y(block_id, i, j, solver)
            
            # corner cells
            self.__apply_north_boundary(block_id, 0, self.mesh.num_cells_y[block_id] - 1, solver)        
            self.__apply_south_boundary(block_id, 0, 0, solver) 
    
    def _north_boundary(self, direction, block_id, solver, var_name):
        if direction == WRT.x:
            for (i, j) in self.mesh.loop_north_bc(block_id, 1):
                self.upwind_wrt_x(block_id, i, j, solver)
            
            # corner cells
            self.__apply_east_boundary(block_id, self.mesh.num_cells_x[block_id] - 1, self.mesh.num_cells_y[block_id] - 1, solver)        
            self.__apply_west_boundary(block_id, 0, self.mesh.num_cells_y[block_id] - 1, solver) 

        elif direction == WRT.y:
            for (i, j) in self.mesh.loop_north_bc(block_id):
                self.__apply_north_boundary(block_id, i, j, solver)          

    def _south_boundary(self, direction, block_id, solver, var_name):
        if direction == WRT.x:
            for (i, j) in self.mesh.loop_south_bc(block_id, 1):
                self.upwind_wrt_x(block_id, i, j, solver)
            
            # corner cells
            self.__apply_east_boundary(block_id, self.mesh.num_cells_x[block_id] - 1, 0, solver)
            self.__apply_west_boundary(block_id, 0, 0, solver) 

        elif direction == WRT.y:
            for (i, j) in self.mesh.loop_south_bc(block_id):
                self.__apply_south_boundary(block_id, i, j, solver)

    def __apply_east_boundary(self, block_id, i, j, solver):
        max_u = max(self.field_manager.fields[pv.velocity_x.name()].picard_old[block_id, i, j], 0.0)
        min_u = min(self.field_manager.fields[pv.velocity_x.name()].picard_old[block_id, i, j], 0.0)

        if self.bc.get_bc_type(block_id, "east") == BCType.dirichlet:
            ap_index = self.mesh.map3Dto1D(block_id, i, j)
            aw_index = self.mesh.map3Dto1D(block_id, i - 1, j)

            ap_value = max_u * self.inv_dx[block_id] - 2.0 * min_u * self.inv_dx[block_id]
            aw_value = - 1.0 * max_u * self.inv_dx[block_id]

            phi = self.bc.get_bc_value(block_id, "east")
            rhs = - 2.0 * phi * min_u * self.inv_dx[block_id]

            solver.add_to_A(ap_index, ap_index, ap_value)
            solver.add_to_A(ap_index, aw_index, aw_value)
            solver.add_to_b(ap_index, rhs)

        elif self.bc.get_bc_type(block_id, "east") == BCType.neumann:
            ap_index = self.mesh.map3Dto1D(block_id, i, j)
            aw_index = self.mesh.map3Dto1D(block_id, i - 1, j)

            ap_value = max_u * self.inv_dx[block_id]
            aw_value = - 1.0 * max_u * self.inv_dx[block_id]

            phi = self.bc.get_bc_value(block_id, "east")
            rhs = - 1.0 * phi * min_u

            solver.add_to_A(ap_index, ap_index, ap_value)
            solver.add_to_A(ap_index, aw_index, aw_value)
            solver.add_to_b(ap_index, rhs)
        
        elif self.bc.get_bc_type(block_id, "east") == BCType.interface:
            self.positive_in_x(block_id, i, j, solver)
            
            neighbour_block_id = self.bc.get_bc_value(block_id, "east")

            ap_index = self.mesh.map3Dto1D(block_id, i, j)
            ae_index = self.mesh.map3Dto1D(neighbour_block_id, 0, j)

            ap_value = -min_u * self.inv_dx[block_id]
            ae_value =  min_u * self.inv_dx[neighbour_block_id]

            solver.add_to_A(ap_index, ap_index, ap_value)
            solver.add_to_A(ap_index, ae_index, ae_value)

    def __apply_west_boundary(self, block_id, i, j, solver):
        max_u = max(self.field_manager.fields[pv.velocity_x.name()].picard_old[block_id, i, j], 0.0)
        min_u = min(self.field_manager.fields[pv.velocity_x.name()].picard_old[block_id, i, j], 0.0)

        if self.bc.get_bc_type(block_id, "west") == BCType.dirichlet:
            ap_index = self.mesh.map3Dto1D(block_id, i, j)
            ae_index = self.mesh.map3Dto1D(block_id, i + 1, j)

            ap_value = 2.0 * max_u * self.inv_dx[block_id] - min_u * self.inv_dx[block_id]
            ae_value = min_u * self.inv_dx[block_id]

            phi = self.bc.get_bc_value(block_id, "west")
            rhs = 2.0 * phi * max_u * self.inv_dx[block_id]

            solver.add_to_A(ap_index, ap_index, ap_value)
            solver.add_to_A(ap_index, ae_index, ae_value)
            solver.add_to_b(ap_index, rhs)

        elif self.bc.get_bc_type(block_id, "west") == BCType.neumann:
            ap_index = self.mesh.map3Dto1D(block_id, i, j)
            ae_index = self.mesh.map3Dto1D(block_id, i + 1, j)

            ap_value = -1.0 * min_u * self.inv_dx[block_id]
            ae_value = min_u * self.inv_dx[block_id]

            phi = self.bc.get_bc_value(block_id, "west")
            rhs = -1.0 * max_u * phi 

            solver.add_to_A(ap_index, ap_index, ap_value)
            solver.add_to_A(ap_index, ae_index, ae_value)
            solver.add_to_b(ap_index, rhs)

        elif self.bc.get_bc_type(block_id, "west") == BCType.interface:
            self.negative_in_x(block_id, i, j, solver)

            neighbour_block_id = self.bc.get_bc_value(block_id, "west")

            ap_index = self.mesh.map3Dto1D(block_id, i, j)
            aw_index = self.mesh.map3Dto1D(neighbour_block_id, self.mesh.num_cells_x[block_id] - 1, j)
            
            ap_value =  max_u * self.inv_dx[block_id]
            aw_value = -max_u * self.inv_dx[neighbour_block_id]

            solver.add_to_A(ap_index, ap_index, ap_value)
            solver.add_to_A(ap_index, aw_index, aw_value)            
    
    def __apply_north_boundary(self, block_id, i, j, solver):
        max_v = max(self.field_manager.fields[pv.velocity_y.name()].picard_old[block_id, i, j], 0.0)
        min_v = min(self.field_manager.fields[pv.velocity_y.name()].picard_old[block_id, i, j], 0.0)

        if self.bc.get_bc_type(block_id, "north") == BCType.dirichlet:
            ap_index = self.mesh.map3Dto1D(block_id, i, j)
            as_index = self.mesh.map3Dto1D(block_id, i, j - 1)

            ap_value = max_v * self.inv_dy[block_id] - min_v * self.inv_dy[block_id]
            as_value = -1.0 * max_v * self.inv_dy[block_id]

            phi = self.bc.get_bc_value(block_id, "north")
            rhs = -2.0 * phi * min_v * self.inv_dy[block_id]

            solver.add_to_A(ap_index, ap_index, ap_value)
            solver.add_to_A(ap_index, as_index, as_value)
            solver.add_to_b(ap_index, rhs)
            
        elif self.bc.get_bc_type(block_id, "north") == BCType.neumann:
            ap_index = self.mesh.map3Dto1D(block_id, i, j)
            as_index = self.mesh.map3Dto1D(block_id, i, j - 1)

            ap_value = max_v * self.inv_dy[block_id]
            as_value = -1.0 * max_v * self.inv_dy[block_id]

            phi = self.bc.get_bc_value(block_id, "north")
            rhs = -1.0 * phi * min_v

            solver.add_to_A(ap_index, ap_index, ap_value)
            solver.add_to_A(ap_index, as_index, as_value)
            solver.add_to_b(ap_index, rhs)

        elif self.bc.get_bc_type(block_id, "north") == BCType.interface:
            self.positive_in_y(block_id, i, j, solver)
            
            neighbour_block_id = self.bc.get_bc_value(block_id, "north")

            ap_index = self.mesh.map3Dto1D(block_id, i, j)
            an_index = self.mesh.map3Dto1D(neighbour_block_id, i, 0)

            ap_value = -min_v * self.inv_dy[block_id]
            an_value =  min_v * self.inv_dy[neighbour_block_id]

            solver.add_to_A(ap_index, ap_index, ap_value)
            solver.add_to_A(ap_index, an_index, an_value)
    
    def __apply_south_boundary(self, block_id, i, j, solver):
        max_v = max(self.field_manager.fields[pv.velocity_y.name()].picard_old[block_id, i, j], 0.0)
        min_v = min(self.field_manager.fields[pv.velocity_y.name()].picard_old[block_id, i, j], 0.0)

        if self.bc.get_bc_type(block_id, "south") == BCType.dirichlet:
            ap_index = self.mesh.map3Dto1D(block_id, i, j)
            an_index = self.mesh.map3Dto1D(block_id, i, j + 1)

            ap_value = 2.0 * max_v * self.inv_dy[block_id] - min_v * self.inv_dy[block_id]
            an_value = min_v * self.inv_dy[block_id]

            phi = self.bc.get_bc_value(block_id, "south")
            rhs = 2.0 * phi * max_v * self.inv_dy[block_id]

            solver.add_to_A(ap_index, ap_index, ap_value)
            solver.add_to_A(ap_index, an_index, an_value)
            solver.add_to_b(ap_index, rhs)

        elif self.bc.get_bc_type(block_id, "south") == BCType.neumann:
            ap_index = self.mesh.map3Dto1D(block_id, i, j)
            an_index = self.mesh.map3Dto1D(block_id, i, j + 1)

            ap_value = -1.0 * min_v * self.inv_dy[block_id]
            an_value = min_v * self.inv_dy[block_id]

            phi = self.bc.get_bc_value(block_id, "south")
            rhs = -1.0 * max_v * phi

            solver.add_to_A(ap_index, ap_index, ap_value)
            solver.add_to_A(ap_index, an_index, an_value)
            solver.add_to_b(ap_index, rhs)

        elif self.bc.get_bc_type(block_id, "south") == BCType.interface:
            self.negative_in_y(block_id, i, j, solver)

            neighbour_block_id = self.bc.get_bc_value(block_id, "south")
            ap_index = self.mesh.map3Dto1D(block_id, i, j)
            as_index = self.mesh.map3Dto1D(neighbour_block_id, i, self.mesh.num_cells_y[block_id] - 1)
            
            ap_value =  max_v * self.inv_dy[block_id]
            as_value = -max_v * self.inv_dy[neighbour_block_id]

            solver.add_to_A(ap_index, ap_index, ap_value)
            solver.add_to_A(ap_index, as_index, as_value)

    def upwind_wrt_x(self, block_id, i, j, solver):
        self.positive_in_x(block_id, i, j, solver)
        self.negative_in_x(block_id, i, j, solver)

    def upwind_wrt_y(self, block_id, i, j, solver):
        self.positive_in_y(block_id, i, j, solver)
        self.negative_in_y(block_id, i, j, solver)

    def positive_in_x(self, block_id, i, j, solver):
        max_u = max(self.field_manager.fields[pv.velocity_x.name()].picard_old[block_id, i, j], 0.0)

        ap_index = self.mesh.map3Dto1D(block_id, i, j)
        aw_index = self.mesh.map3Dto1D(block_id, i - 1, j)
        
        ap_value =  max_u * self.inv_dx[block_id]
        aw_value = -max_u * self.inv_dx[block_id]

        solver.add_to_A(ap_index, ap_index, ap_value)
        solver.add_to_A(ap_index, aw_index, aw_value)

    def negative_in_x(self, block_id, i, j, solver):
        min_u = min(self.field_manager.fields[pv.velocity_x.name()].picard_old[block_id, i, j], 0.0)

        ap_index = self.mesh.map3Dto1D(block_id, i, j)
        ae_index = self.mesh.map3Dto1D(block_id, i + 1, j)

        ap_value = -min_u * self.inv_dx[block_id]
        ae_value =  min_u * self.inv_dx[block_id]

        solver.add_to_A(ap_index, ap_index, ap_value)
        solver.add_to_A(ap_index, ae_index, ae_value)

    def positive_in_y(self, block_id, i, j, solver):
        max_v = max(self.field_manager.fields[pv.velocity_y.name()].picard_old[block_id, i, j], 0.0)

        ap_index = self.mesh.map3Dto1D(block_id, i, j)
        as_index = self.mesh.map3Dto1D(block_id, i, j - 1)

        ap_value =  max_v * self.inv_dy[block_id]
        as_value = -max_v * self.inv_dy[block_id]

        solver.add_to_A(ap_index, ap_index, ap_value)
        solver.add_to_A(ap_index, as_index, as_value)
    
    def negative_in_y(self, block_id, i, j, solver):
        min_v = min(self.field_manager.fields[pv.velocity_y.name()].picard_old[block_id, i, j], 0.0)

        ap_index = self.mesh.map3Dto1D(block_id, i, j)
        an_index = self.mesh.map3Dto1D(block_id, i, j + 1)

        ap_value = -min_v * self.inv_dy[block_id]
        an_value =  min_v * self.inv_dy[block_id]

        solver.add_to_A(ap_index, ap_index, ap_value)
        solver.add_to_A(ap_index, an_index, an_value)
