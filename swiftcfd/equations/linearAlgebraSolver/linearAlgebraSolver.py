from petsc4py import PETSc
from petsc4py import init as petsc_init

from swiftcfd.equations.linearAlgebraSolver.solverFactory import SolverFactory

class LinearAlgebraSolver():
    def __init__(self, params, mesh, var_name):
        # total points in mesh
        self.total_points = mesh.total_points
        self.mesh = mesh
        self.var_name = var_name
        self.is_diagonal = True

        # initialise petsc for usage
        petsc_init()

        # create coefficient matrix A
        self.A = PETSc.Mat().create()
        self.A.setSizes([self.total_points, self.total_points])
        self.A.setType(PETSc.Mat.Type.SEQAIJ)
        self.A.setPreallocationNNZ(9)
        self.A.setUp()

        # create right-hand side vector
        self.b = PETSc.Vec().createSeq(self.total_points)

        # create linear solver
        self.ksp = SolverFactory().create(params, self.var_name)
        self.ksp.setOperators(self.A)        
    
    def reset_A(self):
        self.A.zeroEntries()
    
    def reset_b(self):
        self.b.zeroEntries()

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

    def field_to_petsc_vec(self, field):
        vec = PETSc.Vec().createWithArray(field._data)
        return vec

    def solve(self, field):
        # check if matrix can be inverted trivally (only contains diagonal)
        self.is_diagonal = self.__check_for_diagonal_matrix()

        # convert field to PETSc compatible vector with zero copy
        field_petsc = self.field_to_petsc_vec(field)

        if self.is_diagonal:
            field_petsc.pointwiseDivide(self.b, self.A.getDiagonal())
        else:
            self.ksp.solve(self.b, field_petsc)

    def get_solver_statistics(self):
        num_iterations = self.ksp.getIterationNumber()
        res_norm = self.ksp.getResidualNorm()
        has_converged = self.ksp.getConvergedReason() >= 0
        return self.is_diagonal, num_iterations, res_norm, has_converged

    def __check_for_diagonal_matrix(self):
        number_of_rows, _ = self.A.getSize()
        info = self.A.getInfo(PETSc.Mat.InfoType.LOCAL)
        # nz_used is the "non-zero" entries in the matrix. If there are as many
        # nz entries as there are rows, there is only one entry per row.
        # Thus, the matrix is diagonal and can be trivally inverted for
        # explicit time integration
        return info["nz_used"] == number_of_rows