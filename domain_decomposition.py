import numpy as np
import scipy.sparse as ss
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import multiprocessing as mp

### Finite Difference Methods for Poisson Problems ##################

class Poisson:
    """
    This class contains stores all data needed to solve a Poisson problem

        -u'' = rhs

    with a finite difference method.

    """
    def __init__(self, grid: np.array, h: float, rhs):
        """
        Parameters:

            grid: np.array
                  Grid points including the boundary points.

            h: float
               distance between two grid points.

            rhs: Python function for the right hand side of the
                 differential equation.

        """
        # Grid.
        self.grid = grid

        # Step size of the grid.
        self.h = h

        # Number of grid points. Includes the boundary points
        n = len(grid) - 2

        # The finite difference matrix. Excludes the boundary points.
        A = np.zeros((n, n))
        for i in range(n):
            if i > 0:
                A[i][i-1] = 1
            A[i][i] = -2
            if i < n - 1:
                A[i][i+1] = 1

        A = np.divide(A, -(h**2))

        self.A = ss.csr_matrix(A)

        # Right hand side vector. Excludes the boundary points.
        self.f = np.array([rhs(x) for x in grid[1:-1]])

        # Numpy array with the two boundary values, zero in our case.
        self.boundary_values = np.array([0,0])


# The parts after ':' and '->' are called 'type hints' Python will
# ignore them, but you can better see what input types the function
# expects.
#                                 |                         |         |
#                                 ↓                         ↓         ↓
def set_boundary(poisson_problem: Poisson, boundary_values: np.array) -> Poisson:
        """
        Overrides the boundary conditions of a `Poisson` object.
        This function does not do any particularly interesting,
        it just makes later code a little more convenient.

        Parameters:

            poisson_problem: An instance of `Poisson`.

            boundary_values: np.array
                             Pair of boundary values.

        Returns:

            a `Poisson` object, as the input with new boundary values.

        """
        poisson_problem.boundary_values = boundary_values
        return poisson_problem


def solve(poisson_problem: Poisson) -> np.array:
        """
        Solve a Poisson problem.

        Parameters:

            Poisson: object of type `Poisson`

        Returns

            np.array containing the solution.

        """
        p = poisson_problem

        # Add boundary values to rhs vector
        f = np.array(p.f, copy=True)
        f[0] = f[0] + p.boundary_values[0] / p.h**2
        f[-1] = f[-1] + p.boundary_values[1] / p.h**2

        # Solves Au=F for u
        res = scipy.sparse.linalg.spsolve(p.A, f)

        return res


### Domain Decomposition Function ###################################


def make_sub_grids(grid: np.array, n_domains: int, n_overlap: int) -> [np.array]:
    """
    From one large `grid`, this function computes `n_domains` sub-grids
    that overlap by `n_overlap` grid points.

    Example: n_domains=2, n_overlap=1:


                                  split grid in half -> two grids
                                         ↓
    grid      [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    output[0] [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    output[1]                     [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                                    ↑         ↑
                                    overlap = 1 (each side)

    """
    # Only consider the case the number of intervals between grid
    # points (cells) is divisible by the number of domains.
    n_cells = len(grid) - 1
    assert n_cells % n_domains == 0, f"Number fo cells must be devisable th number of domains but got {n_cells} and {n_domains}"

    # Create subgrids and grab middle indices
    sub_grids = []
    ratio = n_cells / n_domains

    # Use array slicing to create sub_arrays
    for i in range(0, n_domains):
        start = int(end - (2 * n_overlap)) if i != 0 else 0
        end = int(ratio * (i+1)) + n_overlap

        if i == n_domains - 1:
            sub_grids.append(grid[start:])
        else:
            sub_grids.append(grid[start:end+1])

    return sub_grids


def get_dd_shared_values(u: np.array, n_overlap: int) -> np.array:
    """
    Returns the parts of the solution vector that are shared between
    different sub-domains.

    Example:

        grid      [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        neighbour grid                [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        solution       [  1,   2,   3,   4,   5]
        returns                ↑         ↑

    Parameters:

        u: np.array
           solution vector

        n_overlaps: int
                    n_overlaps used earlier to create the sub-domains.

        returns: np.array
                 Array of shape (2,) that contains the boundary values
                 used in neighboring domains by the domain in the
                 decomposition method.


    """
    # Grabs boundary values shared betweeen overlapping domains
    start = n_overlap * 2 - 1
    end = -2 * n_overlap

    return [u[start],u[end]]


def update_dd_boundary(shared_values: np.array) -> np.array:
    """
    Reorders the shared values computed by `get_dd_shared_values` into
    boundary values for each subdomain. You can read of the required
    boundary values by the domain decomposition method formula in the
    project description.

    Example:

        shared values: [[1, 2]
                        [3, 4]
                        [5, 6]]

        return value:  [[0, 3]
                        [2, 5]
                        [4, 0]]


    Parameters:

        shared_values: np.array
                       shared values computed by `get_dd_shared_values`

    Returns:

        A list of boundary values for each domain.

    """
    # set u(0)=u(1)=1
    shared_values[0][0] = shared_values[-1][-1] = 0

    # Flip piece-wise endpoints
    for i in range(len(shared_values) - 1):
        shared_values[i+1][0], shared_values[i][1] = shared_values[i][1], shared_values[i+1][0]

    return shared_values


### Domain Decomposition Solvers ####################################

def solve_dd(problems: [Poisson], n_iter: int, n_overlap: np.array) -> [(np.array, np.array)]:
    """
    Solves the differential equation with a domain decomposition method.

    *** Try to use all the functions above.           ***
    *** Then this function should be reasonably short ***

    Parameters:

        problems: [Poisson]
                  A list of `Poisson` type objects, one for each sub-domain.

        n_iter: int
                Number of iterations conducted by the domain decomposition method.

        n_overlap: int
                number as overlapping cells, same as in other functions above.

    Returns:

        A list [(grid, solution)] of the used (sub-)grid and the corresponding
        solution vector.

    """
    # Iterate m times
    for _ in range(n_iter):
        solutions = []
        shared_values = []

        for p in problems:
            # Solve Au=F
            result = solve(p)
            solutions.append((p.grid, result))

            # Get shared values for neighboring sub-grid
            sv = get_dd_shared_values(result, n_overlap)
            shared_values.append(sv)

        # Get new boundary values from shared values
        boundary_values = update_dd_boundary(shared_values)

        # Update boundary values for each sub-interval
        for k, p in enumerate(problems):
            set_boundary(p, boundary_values[k])

    return solutions


def solver_dd_local(problem: [Poisson], n_overlap: int, queue_in: mp.Queue, queue_out: mp.Queue):
    """
    Solves the local domain differential equation given by `problem`, every time
    it receives new boundary values through `queue_in`. The resulting shared values
    are written into `queue_out`.

    If the `queue_in` contains the string 'Done' this function stops and writes
    the list [grid, solution] of one last solve into the `queue_out`.

    """

    while True:
        boundary_values = queue_in.get()

        if type(boundary_values) == str and boundary_values == 'Done':
            result = solve(problem)
            queue_out.put((problem.grid, result))
            break

        else:
            set_boundary(problem, boundary_values)
            result = solve(problem)
            sv = get_dd_shared_values(result, n_overlap)
            queue_out.put(sv)

def solve_dd_parallel(problems: [Poisson], n_iter: int, n_overlap: int) -> [np.array, np.array]:
    """
    Same as `solve_dd`, except that all sub-domains problems are solved in
    parallel. Use the Python library `multiprocessing` for the implementation.

    """
    processes = []
    queues_in = []
    queues_out = []
    solutions = []
    boundary_values = []

    # Initialize queues and processes with sub-domains
    for p in problems:
        queues_in.append(mp.Queue())
        queues_out.append(mp.Queue())

        process = mp.Process(target=solver_dd_local, args=(p, n_overlap, queues_in[-1], queues_out[-1]))
        processes.append(process)
        process.start()

        # Initialize boundary values array so we can call solver_dd_local
        boundary_values.append(p.boundary_values)

    # Iterate m times, and compute sub-domain results in parallel
    for _ in range(n_iter):
        shared_values = []

        # Write boundary values to queue_in (solve on sub-domain)
        for k, queue_in in enumerate(queues_in):
            queue_in.put(boundary_values[k])

        # Get shared_values from queue_out
        for queue_out in queues_out:
            sv = queue_out.get()
            shared_values.append(sv)

        # Update boundary values with shared values
        boundary_values = update_dd_boundary(shared_values)

    # Complete computation
    for queue_in in queues_in:
        queue_in.put('Done')

    # Get final solutions from sub-domains
    solutions = []
    for queue_out in queues_out:
        solutions.append(queue_out.get())

    return solutions
