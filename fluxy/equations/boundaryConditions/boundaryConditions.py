from enum import Enum, auto

class BCType(Enum):
    dirichlet = auto()
    neumann = auto()
    interface = auto()

class BoundaryConditions():
    def __init__(self, params, mesh, var_name):
        self.mesh = mesh
        self.bc_type = []        
        self.bc_value = []
        self.var_name = var_name

        # add boundary type and value to list
        for index, block in enumerate(range(0, self.mesh.num_blocks)):
            block_id = f'block{index + 1}'

            # bc types and values
            bc_types = {}
            bc_values = {}

            east_type = params.bc(block_id, "east", self.var_name, "type")
            west_type = params.bc(block_id, "west", self.var_name, "type")
            north_type = params.bc(block_id, "north", self.var_name, "type")
            south_type = params.bc(block_id, "south", self.var_name, "type")

            east_value = params.bc(block_id, "east", self.var_name, "value")
            west_value = params.bc(block_id, "west", self.var_name, "value")
            north_value = params.bc(block_id, "north", self.var_name, "value")
            south_value = params.bc(block_id, "south", self.var_name, "value")

            bc_types["east"] = east_type
            bc_types["west"] = west_type
            bc_types["north"] = north_type
            bc_types["south"] = south_type

            bc_values["east"] = float(east_value)
            bc_values["west"] = float(west_value)
            bc_values["north"] = float(north_value)
            bc_values["south"] = float(south_value)

            for key, value in bc_types.items():
                if value == "dirichlet":
                    bc_types[key] = BCType.dirichlet
                elif value == "neumann":
                    bc_types[key] = BCType.neumann
                elif value == "interface":
                    bc_types[key] = BCType.interface
                    bc_values[key] = int(bc_values[key] - 1)            

            self.bc_type.append(bc_types)
            self.bc_value.append(bc_values)

    def apply_boundary_conditions(self, solver, field):
        for block in range(0, self.mesh.num_blocks):
            # east boundary
            if self.bc_type[block]["east"] is not BCType.interface:
                for (i, j) in self.mesh.loop_east(block, 1):
                    index = self.mesh.map3Dto1D(block, i, j)
                    solver.add_to_A(index, index, 1)
                    solver.add_to_b(index, self.get_bc_value(field, i, j, block, "east"))

            # west boundary
            if self.bc_type[block]["west"] is not BCType.interface:
                for (i, j) in self.mesh.loop_west(block, 1):
                    index = self.mesh.map3Dto1D(block, i, j)
                    solver.add_to_A(index, index, 1)
                    solver.add_to_b(index, self.get_bc_value(field, i, j, block, "west"))
            
            # north boundary
            if self.bc_type[block]["north"] is not BCType.interface:
                for (i, j) in self.mesh.loop_north(block):
                    index = self.mesh.map3Dto1D(block, i, j)
                    solver.add_to_A(index, index, 1)
                    solver.add_to_b(index, self.get_bc_value(field, i, j, block, "north"))
            
            # south boundary
            if self.bc_type[block]["south"] is not BCType.interface:
                for (i, j) in self.mesh.loop_south(block):
                    index = self.mesh.map3Dto1D(block, i, j)
                    solver.add_to_A(index, index, 1)
                    solver.add_to_b(index, self.get_bc_value(field, i, j, block, "south"))
    
    def get_bc_value(self, field, i, j, block, face):
        if face == "east":
            if self.bc_type[block]["east"] == BCType.dirichlet:
                return self.bc_value[block]["east"]
            elif self.bc_type[block]["east"] == BCType.neumann:
                dx, _ = self.mesh.get_spacing(block)
                return field.old[block, i - 1, j] + dx * self.bc_value[block]["east"]

        elif face == "west":
            if self.bc_type[block]["west"] == BCType.dirichlet:
                return self.bc_value[block]["west"]
            elif self.bc_type[block]["west"] == BCType.neumann:
                dx, _ = self.mesh.get_spacing(block)
                return field.old[block, i + 1, j] - dx * self.bc_value[block]["west"]

        elif face == "north":
            if self.bc_type[block]["north"] == BCType.dirichlet:
                return self.bc_value[block]["north"]
            elif self.bc_type[block]["north"] == BCType.neumann:
                _, dy = self.mesh.get_spacing(block)
                return field.old[block, i, j - 1] + dy * self.bc_value[block]["north"]

        elif face == "south":
            if self.bc_type[block]["south"] == BCType.dirichlet:
                return self.bc_value[block]["south"]
            elif self.bc_type[block]["south"] == BCType.neumann:
                _, dy = self.mesh.get_spacing(block)
                return field.old[block, i, j + 1] - dy * self.bc_value[block]["south"]
