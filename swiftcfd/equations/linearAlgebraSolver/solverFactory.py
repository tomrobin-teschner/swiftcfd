from petsc4py import PETSc

class SolverFactory():
    def __init__(self):
        pass

    def create(self, params, var_name):
        solver = params.solver('linearSolver', 'solver', var_name)
        preconditioner = params.solver('linearSolver', 'preconditioner', var_name)

        ksp = PETSc.KSP().create()
        ksp.setInitialGuessNonzero(True)

        # set solver``
        if solver == 'RICHARDSON':
            ksp.setType(PETSc.KSP.Type.RICHARDSON)
        elif solver == 'CG':
            ksp.setType(PETSc.KSP.Type.CG)
        elif solver == 'BCGS':
            ksp.setType(PETSc.KSP.Type.BCGS)
        elif solver == 'GMRES':
            ksp.setType(PETSc.KSP.Type.GMRES)
        else:
            print(f'Unknown solver "{solver}" selected')
            print('Available solvers: RICHARDSON, CG, BCGS, GMRES')
            print('Exiting now ...')
            exit(1)
        
        # set preconditioner
        if preconditioner == 'JACOBI':
            ksp.getPC().setType(PETSc.PC.Type.JACOBI)
        elif preconditioner == 'ILU':
            ksp.getPC().setType(PETSc.PC.Type.ILU)
        elif preconditioner == 'SOR':
            ksp.getPC().setType(PETSc.PC.Type.SOR)
        elif preconditioner == 'NONE':
            ksp.getPC().setType(PETSc.PC.Type.NONE)
        else:
            print(f'Unknown preconditioner "{preconditioner}" selected')
            print('Available solvers: JACOBI, ILU, SOR, NONE')
            print('Exiting now ...')
            exit(1)

        # set solver tolerances
        tolerance = params.solver('linearSolver', 'tolerance', var_name)
        max_iterations = params.solver('linearSolver', 'maxIterations', var_name)
        ksp.setTolerances(rtol = tolerance, max_it = max_iterations)

        return ksp