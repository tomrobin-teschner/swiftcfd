from os.path import join, exists
from os import listdir, remove
from os import mkdir

class Output():
    def __init__(self, params, mesh, field_manager):
        self.params = params
        self.mesh = mesh
        self.field_manager = field_manager

        # check content in output folder and remove old solution files
        if exists('output'):
            for file in listdir('output'):
                if file.endswith('.dat'):
                    remove(join('output', file))

    def write(self, iteration = -1):
        case = self.params.solver('output', 'filename')
        filename = case

        if iteration != -1:
            filename += f'_{iteration:010d}.dat' 
        else:
            filename += '.dat'

        if not exists('output'):
            mkdir('output')
        
        with open(join('output', filename), 'w') as f:
            f.write(f'TITLE = "{case}"\n')
            f.write('VARIABLES = "x", "y"')
            for key, _ in self.field_manager.fields.items():
                f.write(f', "{key}"')
            f.write('\n')

            for block in range(0, self.mesh.num_blocks):
                f.write(f'\nZONE T="Block{block + 1}", I={self.mesh.num_x[block]}, J={self.mesh.num_y[block]}, F=POINT\n')
                for j in range(0, self.mesh.num_y[block]):
                    for i in range(0, self.mesh.num_x[block]):
                        f.write(f'{self.mesh.x[block][i][j]} {self.mesh.y[block][i][j]}')
                        for _, value in self.field_manager.fields.items():
                            f.write(f' {value[block, i, j]}')
                        f.write('\n')
