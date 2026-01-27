from petsc4py import PETSc

class SolverFactory():
    def __init__(self):
        pass

    def create(self, params, var_name):
        solver = params('solver', 'linearSolver', 'solver', var_name)
        preconditioner = params('solver', 'linearSolver', 'preconditioner', var_name)

        ksp = PETSc.KSP().create()
        ksp.setInitialGuessNonzero(True)

        # set solver
        # available solver types: https://petsc.org/release/petsc4py/reference/petsc4py.PETSc.KSP.Type.html#petsc4py.PETSc.KSP.Type
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
        # available preconditioners: https://petsc.org/release/petsc4py/reference/petsc4py.PETSc.PC.Type.html
        if preconditioner == 'JACOBI':
            ksp.getPC().setType(PETSc.PC.Type.JACOBI)
        elif preconditioner == 'ILU':
            ksp.getPC().setType(PETSc.PC.Type.ILU)
        elif preconditioner == 'SOR':
            ksp.getPC().setType(PETSc.PC.Type.SOR)
        elif preconditioner == 'GAMG':
            ksp.getPC().setType("gamg")
            opts = PETSc.Options()
            opts["pc_gamg_type"] = "agg"
            opts["pc_gamg_threshold"] = 0.01
            opts["pc_gamg_square_graph"] = 1
            # opts["mg_levels_ksp_type"] = "chebyshev"
            # opts["mg_levels_pc_type"] = "jacobi"
            opts["mg_levels_ksp_type"] = "richardson"
            opts["mg_levels_pc_type"]  = "jacobi"
            # opts["mg_levels_pc_type"] = "ilu"
            # opts["mg_levels_pc_factor_levels"] = 0
            opts["mg_levels_ksp_max_it"] = 3
            opts["mg_coarse_ksp_type"] = "preonly"
            opts["mg_coarse_pc_type"] = "lu"

            ksp.setFromOptions()
        elif preconditioner == 'NONE':
            ksp.getPC().setType(PETSc.PC.Type.NONE)
        else:
            print(f'Unknown preconditioner "{preconditioner}" selected')
            print('Available solvers: JACOBI, ILU, SOR, GAMG, NONE')
            print('Exiting now ...')
            exit(1)

        # set solver tolerances
        tolerance = params('solver', 'linearSolver', 'tolerance', var_name)
        max_iterations = params('solver', 'linearSolver', 'maxIterations', var_name)
        ksp.setTolerances(rtol = tolerance, max_it = max_iterations)

        # ksp.setComputeSingularValues(True)

        return ksp