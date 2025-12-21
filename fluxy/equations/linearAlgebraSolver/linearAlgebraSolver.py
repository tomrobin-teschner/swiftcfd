from petsc4py import PETSc
from petsc4py import init as petsc_init

class LinearAlgebraSolver():
    def __init__(self, params, mesh):
        # total points in mesh
        total_points = mesh.total_points

        # initialise petsc for usage
        petsc_init()

        # create coefficient matrix A
        self.A = PETSc.Mat().create()
        self.A.setSizes([total_points, total_points])
        self.A.setType(PETSc.Mat.Type.SEQAIJ)
        self.A.setUp()
        self.A.setPreallocationNNZ(3)

        # create right-hand side vector
        self.b = PETSc.Vec().createSeq(total_points)

        # create linear solver
        self.ksp = PETSc.KSP().create()
        self.ksp.setType(PETSc.KSP.Type.GMRES)
        self.ksp.getPC().setType(PETSc.PC.Type.JACOBI)
        self.ksp.setTolerances(rtol=1e-8, atol=1e-12, divtol=1e5, max_it=1000)
    
    def reset_A(self):
        self.A.zeroEntries()
        self.A.assemble()
    
    def reset_b(self):
        self.b.zeroEntries()
        self.b.assemble()

    def add_to_A(self, row, col, value):
        self.A.setValue(row, col, value, addv=PETSc.InsertMode.ADD_VALUES)

    def insert_into_A(self, row, col, value):
        self.A.setValue(row, col, value, addv=PETSc.InsertMode.INSERT_VALUES)
    
    def add_to_b(self, row, value):
        self.b.setValue(row, value, addv=PETSc.InsertMode.ADD_VALUES)

    def insert_into_b(self, row, value):
        self.b.setValue(row, value, addv=PETSc.InsertMode.INSERT_VALUES)

    def assemble(self):
        self.A.assemblyBegin()
        self.A.assemblyEnd()

        self.b.assemblyBegin()
        self.b.assemblyEnd()

    def solve(self, x):
        self.assemble()
        self.ksp.setOperators(self.A)
        self.ksp.solve(self.b, x)
    
    def get_solver_statistics(self):
        num_iterations = self.ksp.getIterationNumber()
        res_norm = self.ksp.getResidualNorm()
        has_converged = self.ksp.getConvergedReason() >= 0
        return num_iterations, res_norm, has_converged