from os. path import join
import toml

class Parameters:
    """reads parameters from TOML parameter files"""
    def __init__(self):
        self.params = {}
        # temp_parameters = self._get_parameters(join(root, 'config.toml'))
        # case_name = temp_parameters['inputFiles']['case']

        # self.solver_params = self._get_parameters(join(root, 'solver', case_name + '.toml'))
        # self.mesh_params = self._get_parameters(join(root, 'mesh', case_name + '.toml'))
        # self.bc_params = self._get_parameters(join(root, 'boundaryConditions', case_name + '.toml'))

    def read_from_file(self, filename):
        self.params = self._get_parameters(filename)
        self._check_parameters()
    
    def read_from_string(self, string):
        self.params = toml.loads(string)
        self._check_parameters()

    def _get_parameters(self, filename):
        with open(filename) as f:
            return toml.load(f)

    def _check_parameters(self):
        mesh_blocks = len(self.params['mesh'])
        bc_blocks = len(self.params['boundaryCondition'])
        assert(mesh_blocks == bc_blocks), 'number of mesh and boundary condition blocks must be equal'

    def __call__(self, *args):
        temp = self.params
        for key in args:
            temp = temp[key]
        return temp

    # def solver(self, *args):
    #     temp = self.solver_params
    #     for key in args:
    #         temp = temp[key]
    #     return temp

    # def mesh(self, *args):
    #     temp = self.mesh_params
    #     for key in args:
    #         temp = temp[key]
    #     return temp

    # def bc(self, *args):
    #     temp = self.bc_params
    #     for key in args:
    #         temp = temp[key]
    #     return temp

    # def num_points(self):
    #     self.num_points = 0 
    #     for i in range(0, self.num_blocks):
    #         block = f'block{i + 1}'
    #         num_x = self.mesh(block, 'x', 'num')
    #         num_y = self.mesh(block, 'y', 'num')
    #         self.num_points += num_x * num_y