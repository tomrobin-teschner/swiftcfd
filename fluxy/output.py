from os.path import join, exists
from os import mkdir

class Output():
    def __init__(self, params, mesh):
        self.params = params
        self.mesh = mesh

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
            f.write('VARIABLES = "x", "y"\n')
            # for id in self.params.solver('output', 'IDs'):
            #     f.write(f', "{self.params.solver("output", "pvNames", id)}"')
            # f.write('\n')

            for block in range(0, self.mesh.num_blocks):
                f.write(f'\nZONE T="Block{block + 1}", I={self.mesh.num_x[block]}, J={self.mesh.num_y[block]}, F=POINT\n')
                for j in range(0, self.mesh.num_y[block]):
                    for i in range(0, self.mesh.num_x[block]):
                        f.write(f'{self.mesh.x[block][i][j]} {self.mesh.y[block][i][j]}\n')
