![If you can't see this, imagine this to be the best logo you have ever seen, at least considering it was thrown together in 5 minutes](logo/logo_400.png)

![Static Badge](https://img.shields.io/badge/Version-0.33.0-blue)

> [!WARNING]  
> This project is work in progress and in version 0.X, expect breaking changes with new features.

Swiftcfd is a 2D solver for cartesian, block-structured grids for fully implicit, incompressible Navier-Stokes simulations. It has been designed around an ```Equation``` class that allows to quickly prototype new equations and test these with common Krylov subspace methods to solve the linear system $\mathbf{Ax}=\mathbf{b}$.

In addition, various deep neural networks have been implemented to inject an initial solution into $\mathbf{Ax}=\mathbf{b}$ to aide convergence. This is a work-in-progress and should be treated as such. Additional information on the combined ML and CFD can be found at the end of this README.md file in the section titled ```Using the ML module of swiftcfd```.

## Installation

This CFD solver requires ```PETSc``` to run, which can only be compiled on UNIX. Either a UNIX environment (Linux / macOS) is required, or [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) on Windows. Since ```PETSc``` is compiled from source, you will need a basic development environment installed on your PC (Python3, a C++/C compiler, and autotools).

### Docker and docker compose

The simplest way to get up and running is to use Docker. It works on any operating system, and since everythign has been preconfigured, you shouldn't see any nasty surprises trying to get this up and running. The only prerequisite is that you have docker installed. If you have that, you can simply type:

```bash
docker compose build swiftcfd
```

If you are on UNIX, you may need ```sudo``` priviliges to execute Docker, unless you have given special priviliges to Docker. When you run this command for the first time, expect this to take about an hour to configure. Afterwards the execution will be instant. You can change the input file, more on that at the end of this README file.

### UNIX (WSL) installation

You can also use a UNIX (or WSL shell on Windows) if you prefer. If you use that instead, you can manually configure and install PETSc. First, we need to ensure we have sensible default build tools available:

```bash
sudo update
sudo apt install -y build-essential
```

Once the development environment is set up, you should be able to install all Python packages with the following commands:

```bash
export PETSC_CONFIGURE_OPTIONS="--download-fblaslapack=1"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The installation will download and compile PETSc, which can take a good 10 minutes or so, without any output being shown to the screen. If the installation fails, an error message will be printed.

When closing the terminal and opening it again, you will need to load the virtual environment again using:

```bash
source .venv/bin/activate
```

## Usage

To run the solver, an input file is required. Sample input files are shown in the ```input/``` folder, which all use the [TOML](https://toml.io/en/) file extension and formatting. These input files have different section pertaining to the solver settings, the meshing, and the boundary conditions.

To execute the solver with a specific input file, use the following command:

```bash
docker compose run --rm swiftcfd python3 swiftcfd.py --input input/INPUT_FILE.toml
```

If you have installed directly into WSL or a UNIX shell, simply use:

```bash
python3 swiftcfd.py --input input/INPUT_FILE.toml
```

Here, replace ```INPUT_FILE``` by an appropriate filename, e.g. ```heatedCavitySingle.toml```, where the ```Single``` indicates that a single block is used.

### Implemented solvers (equations)

Looking into the input files, we can see that the following solvers (equations) are implemented:

```TOML
[solver.equation]
# Specify the solver to use. Available solvers are:
# heatDiffusion, pressureProjection
solver = 'heatDiffusion'
```

The ```heatDiffusion``` solver solves the 2D unsteady, implicit heat equation of the form:

$$
\frac{\partial T}{\partial t}=\alpha\left(\frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2}\right)
$$

The ```pressureProjection``` algorithm solves the 2D unsteady Navier-Stokes equations using Chorin's exact projection method. Here, we first solve the momentum equations in $x$ and $y$ as:

$$
\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y} = \nu\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right)
$$

$$
\frac{\partial v}{\partial t} + u\frac{\partial v}{\partial x} + v\frac{\partial v}{\partial y} = \nu\left(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2}\right)
$$

All terms are evaluated at time level $n+1$, that is, the momentum equations are solved in a fully implicit manner. Since this equation is non-linear, and we are using a linear system of equation solver here, we need to restore the non-linear behaviour through a correction. This is done through Picard iterations, and the number of non-linear correction steps/iterations can be specified in the input file with:

```TOML
[solver.convergence]
# number of picard iterations
picardIterations = 1
```

We don't have to correct the linearisation, but it may lead to better time accurate results.

Once the momentum equatiosn have been solved, and potentially corrected with our Picard iterations, we solve the pressure Poisson equation of the form:

$$
\frac{\partial^2 p}{\partial x^2} + \frac{\partial^2 p}{\partial y^2} = \frac{\rho}{\Delta t} \left( \frac{\partial u^{n+1/2}}{\partial x} + \frac{\partial v^{n+1/2}}{\partial y} \right)
$$

Here, $u^{n+1/2}$ and $v^{n+1/2}$ are the predicted (intermediate) velocities from the momentum equations. The pressure is evaluated at $n+1$ (implicitly), while the right-hand side is evaluated fully explicitly. With the pressure available at $n+1$, we can correct the velocities through:

$$
u^{n+1}=u^{n+1/2} - \frac{\Delta t}{\rho}\frac{\partial p}{\partial x}
$$

$$
v^{n+1}=v^{n+1/2} - \frac{\Delta t}{\rho}\frac{\partial p}{\partial y}
$$

### Linear solvers

The implicit equations are brought into the form of $\mathbf{Ax}=\mathbf{b}$ and are then subsequently solved with PETSc. The solver to use for each equation can be done in the ```solver.linearSolver``` node under the ```solver``` variable. An example is shown below:

```TOML
[solver.linearSolver]
# Linear solver for the implicit treatment. Available solvers are:
# RICHARDSON, CG, BCGS, GMRES
solver = {u = "BCGS", v="BCGS", p="BCGS"}
```

The following solvers are available:

- ```RICHARDSON```: Baseline (Richardson) method for solving linear systems.
- ```CG```: Conjugate Gradient, for symmetric matrices only (pressure and temperature)
- ```BCGS```: Bi-Conjugate Gradient Stabilised, for symemtric and non-symmetric matrices (velocities, mostly)
- ```GMRES```: Generalised Minimal Residual, for symmetric and non-symmetric matrices (velocities, mostly)

We have to specify which equation uses which solver. The following name convention holds:

- ```u```: u-momentum equation
- ```v```: v-momentum equation
- ```p```: pressure Poisson solver
- ```T```: temperature (heat diffusion)

We can also apply preconditioning, as shown below:

```TOML
# Preconditioner for the implicit treatment. Available preconditioners are:
# JACOBI, ILU, SOR, GAMG, NONE
preconditioner = {u = "ILU", v="ILU", p="ILU"}
```

The following preconditioners are available:

- ```JACOBI```: Jacobi preconditioner (divide by main diagonal of $\mathbf{A}$)
- ```ILU```: Incomplete LU decomposition
- ```SOR```: Gauss-Seidel with Successive Overrelaxation (SOR)
- ```GAMG```: Geometric Agglomerated Algebraic Multigrid
- ```NONE```: Well, no preconditioner

We can specify the maximum number of iterations for each solver to use, as well as the convergence criterion a shown in the following:

```TOML
# relative tolerance to reach for the linear solver within each time step
tolerance = {u = 1e-10, v = 1e-10, p = 1e-10}

# max allowable iterations by the implicit solver
maxIterations = {u = 1000, v = 1000, p = 1000}
```

Here, we first compute $residual = \mathbf{Ax}-\mathbf{b}$ (since $\mathbf{Ax}=\mathbf{b}$, i.e. if we have found a valid solution, $\mathbf{Ax}-\mathbf{b}=0$). If $residual$ is smaller than the value given by ```tolerance``` for each variable, then the linear solver converges and goes to the next time step.

There are two more tolerances, which defined under the ```solver.convergence``` node. These are shown below:

```TOML
[solver.convergence]
# number of picard iterations
picardIterations = 10

# residual tolerance for picard iteration convergence
picard_tolerance = {u = 1e-3, v = 1e-3}

# residual tolerance for the overall simulation
convergence_tolerance = {u = 1e-8, v = 1e-8, p = 1e-8}
```

We already saw the ```picardIterations```. The ```picard_tolerance``` tells the linear solver to stop correcting the non-linear term if the difference between the variables (here ```u``` and ```v```) between subsequent Picard iterations is less than the tolerance specified.

The ```convergence_tolerance``` tells the solver to stop completely once these convergence thresholds have been met by each equation. Thus, to compare all available convergence metrics, this is what each is doing:

- ```tolerance``` is met: We go to the next Picard iteration. If we only have 1 Picard iteration, then we go to the next time step.
- ```picard_tolerance``` is met: We go to the next time step.
- ```convergence_tolerance``` is met: We stop the solver and write out the solution.

### Implemented numerical schemes

The numerical schemes are set in the ```solver.schemes``` node. An example is shown below:

```TOML
[solver.schemes]
# time-integration scheme to use. Available schemes are:
# firstOrderEuler, secondOrderBackwards
timeIntegrationScheme = 'secondOrderBackwards'

# non-linear scheme to use. Available schemes are:
# firstOrderUpwind, secondOrderUpwind
nonLinearScheme = 'secondOrderUpwind'

# diffusion scheme to use. Available schemes are:
# secondOrderCentral
diffusionScheme = 'secondOrderCentral'
```

The following schemes can be selected:

#### Time discretisation

- ```firstOrderEuler```: Integrates the time derivative with a first-order time accurate algorithm:

$$
\frac{\partial \phi}{\partial t}\approx \frac{\phi^{n+1}_{i,j}-\phi^{n}_{i,j}}{\Delta t}
$$

- ```secondOrderBackwards```: Integrates the time derivative with a second-order time accurate algorithm:

$$
\frac{\partial \phi}{\partial t}\approx \frac{3\phi^{n+1}_{i,j}-4\phi^{n}_{i,j}+\phi^{n-1}_{i,j}}{2\Delta t}
$$

#### First-order derivatives

```firstOrderUpwind```: Implements the first-order accurate upwind scheme as:

$$
u\frac{\partial \phi}{\partial x}\approx \text{max}(u,0)\frac{\phi_{i,j}-\phi_{i-1,j}}{\Delta x} + \text{min}(u,0)\frac{\phi_{i+1,j}-\phi_{i,j}}{\Delta x}
$$

$$
v\frac{\partial \phi}{\partial y}\approx \text{max}(u,0)\frac{\phi_{i,j}-\phi_{i-1,j}}{\Delta y} + \text{min}(u,0)\frac{\phi_{i+1,j}-\phi_{i,j}}{\Delta y}
$$

```secondOrderUpwind```: Implements the second-order accurate upwind scheme as:

$$
u\frac{\partial \phi}{\partial x}\approx \text{max}(u,0)\frac{3\phi_{i,j}-4\phi_{i-1,j}+\phi_{i-2,j}}{2\Delta x} + \text{min}(u,0)\frac{-3\phi_{i,j}+4\phi_{i+1,j}-\phi_{i+2,j}}{\Delta x}
$$

$$
v\frac{\partial \phi}{\partial y}\approx \text{max}(v,0)\frac{3\phi_{i,j}-4\phi_{i-1,j}+\phi_{i-2,j}}{2\Delta y} + \text{min}(v,0)\frac{-3\phi_{i,j}+4\phi_{i+1,j}-\phi_{i+2,j}}{\Delta y}
$$

#### Second-order derivatives

```secondOrderCentral```: Implements second-order derivatives (diffusion-type derivatives) using a second-order accurate approximation as:

$$
\frac{\partial^2 \phi}{\partial x^2}\approx \frac{\phi_{i+1,j}-2\phi_{i,j}+\phi_{i-1,j}}{(\Delta x)^2}
$$

The second-order central differencing scheme is also implemented for explicit discretisations, though currently not used. In this case, the derivative is evaluated based on known quantities at time level $n$ and it will be absorbed into the right-hand side vector $\mathbf{b}$ in $\mathbf{Ax}=\mathbf{b}$.

### Meshing

```swiftcfd``` comes with a simple block-structured mesh generator. An example input for a mesh with 2 blocks is shown below:

```TOML
[mesh.block1]
# start/end location and number of cells to use in the x direction
x = {start = 0.0, end = 10.0, numCells = 50}

# start/end location and number of cells to use in the y direction
y = {start = 0.0, end = 0.5, numCells = 10}

[mesh.block2]
# start/end location and number of cells to use in the x direction
x = {start = 0.0, end = 10.0, numCells = 50}

# start/end location and number of cells to use in the y direction
y = {start = 0.5, end = 1.0, numCells = 10}
```

For each block, we specify the ```start``` and ```end``` of the ```x``` and ```y``` coordinates within each block. We also specify how many cells we want to use in each direction.

We can create as many blocks as we want. To tell the solver which blocks share an interface, we impose special ```interface``` boundary condition.

### Boundary conditions

Boundary conditions are imposed for each block. Each block will have an ```east```, ```west```, ```north```, and ```south``` face, and for each face, we impose boundary conditions for each variable. An example for the same 2 block mesh shown above is given below:

```TOML
[boundaryCondition.block1]
# boundary conditions in the east direction. Available boundary conditions are:
# dirichlet, neumann, interface
east =  {u = {type = "neumann", value = 0.0}, v = {type = "neumann", value = 0.0}, p = {type = "dirichlet", value = 0.0}}

# boundary conditions in the west direction. Available boundary conditions are:
# dirichlet, neumann, interface
west =  {u = {type = "dirichlet", value = 1.0}, v = {type = "dirichlet", value = 0.0}, p = {type = "neumann", value = 0.0}}

# boundary conditions in the north direction. Available boundary conditions are:
# dirichlet, neumann, interface
north = {u = {type = "interface", value = 2}, v = {type = "interface", value = 2}, p = {type = "interface", value = 2}}

# boundary conditions in the south direction. Available boundary conditions are:
# dirichlet, neumann, interface
south = {u = {type = "dirichlet", value = 0.0}, v = {type = "dirichlet", value = 0.0}, p = {type = "neumann", value = 0.0}}

[boundaryCondition.block2]
# boundary conditions in the east direction. Available boundary conditions are:
# dirichlet, neumann, interface
east =  {u = {type = "neumann", value = 0.0}, v = {type = "neumann", value = 0.0}, p = {type = "dirichlet", value = 0.0}}

# boundary conditions in the west direction. Available boundary conditions are:
# dirichlet, neumann, interface
west =  {u = {type = "dirichlet", value = 1.0}, v = {type = "dirichlet", value = 0.0}, p = {type = "neumann", value = 0.0}}

# boundary conditions in the north direction. Available boundary conditions are:
# dirichlet, neumann, interface
north = {u = {type = "dirichlet", value = 0.0}, v = {type = "dirichlet", value = 0.0}, p = {type = "neumann", value = 0.0}}

# boundary conditions in the south direction. Available boundary conditions are:
# dirichlet, neumann, interface
south = {u = {type = "interface", value = 1}, v = {type = "interface", value = 1}, p = {type = "interface", value = 1}}
```

For each face and each variable, we have to specify a type and a value. The following types are allowed:

- ```dirichlet```
- ```neumann```
- ```interface```

Dirichlet and Neumann boundary condition impose a fixed value or a flux, repsectively. The fixed value (of flux) is imposed through the ```value``` property for each boundary condition.

The ```interface``` boundary condition tells the solver that there is another block on the other side of this face. The value that is specified is the ID of the block on the other side.

For example, in the case given above, we have two long channels on top of each other, with the first block on the bottom, and the second block above the first block. Therefore, the ```north``` boundary condition for the first block states that we have an ```interface```, and it is connected to the second block (```value``` is equal to 2). Equally, the second block has an ```interface``` at its south boundary, with its ```value``` equal to 1, as it points to the first block.

To impose common boundary conditions, use the following combinations:

- Wall: Neumann for pressure, Dirichlet for both velocities with a value of zero (or non-zero for moving walls).
- Velocity inlet: Dirichlet for both velocities (imposing a specific inlet velocity), Neumann for pressure
- Pressure outlet: Dirichlet for pressure, Neumann for velocities (zero gradient, i.e. ```value``` should be zero)
- Symmetry: All quantities have Neumann boundary conditions

## Available cases

```swiftcfd``` comes with a number of default cases which you can run to get an idea for how to use the solver, what output is generated. The following cases are available:

- ```bfs.toml```: Flow over a backward facing step (bfs)
- ```channel.toml```: Simple flow inside a rectangular channel
- ```channel2blocks.toml```: Essentially the same as the channel case before, just with an interface at the centerline (we looked at that in the meshing and boundary condition discussion above)
- ```cylinder.toml```: The flow around a square cylinder with von Karamn vortex shedding
- ```heatedCavity.toml```: A simple square cavity solving the heat diffusion equation using several blocks
- ```heatedCavitySingle.toml```: Same as above, just with a single block
- ```heatedChannel.toml```: A channel with an obstacle in the center with different heating applied at the top and bottom boundary.

To run any of these case, use the command we already saw:

```bash
python3 swiftcfd.py --input input/INPUT_FILE.toml
```

and replace ```INPUT_FILE.toml``` by any of the cases shown above. After the simulation has finished, you will get a new output folder within the ```output``` folder, e.g. ```output/channel```, which will contain the following items:

- ```<case-name>.dat```: This is the solution file for your case. You can read this in etiehr Tecplot or paraview. If you read this in paraview, it is recommended to apply the following two filters, in this order: *Merge Blocks* followed by *Cell Data to Point Data*. This will produces interpolated contour plots for a better representation.
- ```<case-name>_statistics.dat```: Some statistics about the performance of the linear solver.
- ```<case-name>_000001.dat```: If you have selected to write out an animation (the variable ```writingFrequency``` is not set to zero but instead an interval at which solution animation files shoudl be written), then these files fill be visible as well. If ```writingFrequency``` is equal to zero, these files will not be created.
- ```contours.png```: A contour plot of the final solution.
- ```residuals.png```: Residual plot for the simulation.
- ```residuals.csv```: The residuals used for the plot above, can be used for additional processing.
- ```trainingData_<variable>.csv```: If we have set the variable ```generateTrainingData``` to ```true```, then we will collect variables in a specific format, written as a ```*.csv``` file, which can be used for Machine Learning training of this solver.

## Using the ML module of swiftcfd

```swiftcfd``` has native support for convergence acceleration through PyTorch using a modified (numerical) Physics-Informed Neural Network (PINN) approach. This implementation is currently experimental. To generate training data, run as many test cases as you wish to create training data. E.g.:

```bash
docker compose run --rm swiftcfd python3 swiftcfd.py --input input/channel.toml
```

Check that the ```generateTrainingData``` variable within the input file (e.g. ```input/channel.toml```) is set to ```true```, and that the variables that should be written to file are set in the input file as well. For example, to write out trainign data for the ```u```, and ```v``` velocity, as well as for the pressure ```p```, your input file should contain the following section:

```toml
[solver.ML]
# generate training data
generateTrainingData = true

# variables to output
trainingVariables = ['u', 'v', 'p']
```

Repeat this for as many test cases as you wish. Once all test cases have been recorded, you can train the network using the following command:

```bash
docker compose run --rm swiftcfd python3 swiftcfd.py --train --model=mlp --variables=u,v,p
```

Here, we specify that we want to train the model, whcih bypasses the CFD solver entirely and only trains the network based on the variables we provde, in this case ```u```, ```v``` and ```p``` (we have to provide the variables here again in case additional data sets exist for other variables).

We also have to provide the model we wish to train on, and we have ```mlp```, ```rnn```, ```lstm```, and ```transformer``` available. If the ```--train``` flag is provided without the ```--model``` and ```--variables``` flag, the solver will not continue and crash.