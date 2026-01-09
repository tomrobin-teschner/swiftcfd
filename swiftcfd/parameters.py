from os. path import join
import toml

class Parameters:
    """reads parameters from TOML parameter files"""
    def __init__(self):
        self.params = {}

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