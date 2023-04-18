# Domain Decomposition Finite Difference Method for Poisson Equation

This repository contains a Python implementation of the domain decomposition finite difference method applied to the Poisson equation. The implementation was completed as a final project for a numerical methods in differential equations course.

## Description

The project solves the Poisson problem -u''(x) = f(x) with given boundary conditions, using the finite difference method and domain decomposition. The code consists of several functions for creating sub-domains, setting boundary conditions, solving the Poisson problem, and updating shared values between overlapping sub-domains. The implementation also includes a parallelized version of the domain decomposition solver using Python's multiprocessing library.

The main script demonstrates the usage of these functions, solving a sample Poisson problem with **'f(x) = 1, u(0) = u(1) = 0'**, and comparing the local and parallel solutions.

## Dependencies

   - numpy
   - scipy
   - matplotlib
   - multiprocessing

## How to Use

    1. Clone the repository.
    2. Create a new python file, define your functions (see example below), and run the file.  Alternatively, run unit_tests.py to see passing results.
  
## Example


```python

import numpy as np
from domain_decomposition import *

# Define the right-hand side function
def f(x):
    return 1

n_domains = 4
n_overlap = 4
n_iter = 10
cells = 20

a = 0
b = 1

# Create grid and subgrids
grid = np.linspace(a, b, n_domains*cells + 1)
sub_grids = make_sub_grids(grid, n_domains, n_overlap)

h = 1 / (len(grid) - 1)
rhs = f

problems = []

# Create Poisson problems for each subgrid
for i in range(n_domains):
    problems.append(Poisson(sub_grids[i], h, rhs))

# Solve using domain decomposition
local_res = solve_dd(problems, n_iter, n_overlap)
parallel_res = solve_dd_parallel(problems, 1, n_overlap)'
```
