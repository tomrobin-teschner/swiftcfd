## Installation

This CFD solver requires ```PETSc``` to run, which can only be compiled on UNIX. Either a UNIX environment (Linux / macOS) is required, or [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) on Windows.

The following lines will install the requirements to run this solver:

```bash
export PETSC_CONFIGURE_OPTIONS="--download-fblaslapack=1"
python3 -m venv .venv
source .venv/bin/activate
.venv/bin/python3 -m pip install -r requirements.txt
```

