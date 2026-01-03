# swiftcfd

![Static Badge](https://img.shields.io/badge/Version-0.15.1-blue)

A library to quickly prototype 2D CFD solvers using the finite difference method on a Cartesian grid and implicit time integration.

## Installation

This CFD solver requires ```PETSc``` to run, which can only be compiled on UNIX. Either a UNIX environment (Linux / macOS) is required, or [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) on Windows. I suppose Cygwin would do as well, but why bother if WSL exists?

The following lines will install the requirements to run this solver:

```bash
export PETSC_CONFIGURE_OPTIONS="--download-fblaslapack=1"
python3 -m venv .venv
source .venv/bin/activate
.venv/bin/python3 -m pip install -r requirements.txt
```
The installation will download and compile PETSc, which can take a good 10 minutes or so, without any output being shown to the screen.

If you are getting error messages for the above installation steps, ensure common development tools are available. In particular, you will need a C/C++ compiler, as well as autotools (argh).

On Ubuntu, for example, install the essential build tools as:

```bash
sudo update
sudo apt install -y build-essential
```

