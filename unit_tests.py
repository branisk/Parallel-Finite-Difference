import unittest
import hashlib
import os
from domain_decomposition import *


class TestFiniteDifference(unittest.TestCase):

    def setUp(self):

        self.rhs = lambda x: x*x

        self.grid = np.linspace(0, 1, 11)
        self.h = 1 / (len(self.grid) - 1)

        self.problem = Poisson(self.grid, self.h, self.rhs)


    def test_h(self):

        np.testing.assert_almost_equal(self.h, 0.1)

    def test_A(self):

        np.testing.assert_array_almost_equal(
            self.problem.A.todense()[2,0:5],
            np.array([[0, -100, 200, -100, 0]]))

    def test_f(self):

        np.testing.assert_almost_equal(self.problem.f[0], 0.1**2)
        np.testing.assert_almost_equal(self.problem.f[1], 0.2**2)

    def test_set_boundary(self):

        boundary_values = np.array([1,2])
        problem = set_boundary(self.problem, (1,2))
        result = np.asarray(problem.boundary_values)

        np.testing.assert_array_almost_equal(result, boundary_values)

    def test_solve(self):

        problem = set_boundary(self.problem, np.array([1,2]))
        expected = np.array([1.10825, 1.2164, 1.32415, 1.431, 1.53625, 1.639, 1.73815, 1.8324, 1.92025])
        solution = solve(self.problem)
        np.testing.assert_array_almost_equal(solution, expected, decimal=3)


class TestDomainDecompositionComponents(unittest.TestCase):

    def setUp(self):

        pass

    def test_subgrids_two_domains_overlap_one(self):

        grid = np.linspace(0, 1, 11)
        sub_grids = make_sub_grids(grid, n_domains=2, n_overlap=1)
        self.assertEqual(len(sub_grids[0]), 7)
        self.assertEqual(len(sub_grids[1]), 7)
        np.testing.assert_almost_equal(sub_grids[0][-1], 0.6)
        np.testing.assert_almost_equal(sub_grids[1][0], 0.4)

    def test_subgrids_two_domains_overlap_two(self):

        grid = np.linspace(0, 1, 11)
        sub_grids = make_sub_grids(grid, n_domains=2, n_overlap=2)
        self.assertEqual(len(sub_grids[0]), 8)
        self.assertEqual(len(sub_grids[1]), 8)
        np.testing.assert_almost_equal(sub_grids[0][-1], 0.7)
        np.testing.assert_almost_equal(sub_grids[1][0], 0.3)

    def test_subgrids_four_domains_overlap_one(self):

        grid = np.linspace(0, 1, 21)
        sub_grids = make_sub_grids(grid, n_domains=4, n_overlap=1)
        self.assertEqual(len(sub_grids[0]), 7)
        self.assertEqual(len(sub_grids[1]), 8)
        self.assertEqual(len(sub_grids[2]), 8)
        self.assertEqual(len(sub_grids[3]), 7)
        np.testing.assert_almost_equal(sub_grids[0][-1], 0.3)
        np.testing.assert_almost_equal(sub_grids[2][0], 0.45)

    def test_shared_values(self):

        grid = np.linspace(0, 1, 11)

        # 2 domains, overplap 1
        sub_grids = make_sub_grids(grid, n_domains=2, n_overlap=1)
        u_left = sub_grids[0][1:-1] # for u(x) = x, boundary points not included.
        u_right = sub_grids[1][1:-1] # for u(x) = x, boundary points not included.
        sv = get_dd_shared_values(u_left, n_overlap=1)
        np.testing.assert_array_almost_equal(sv, [0.2, 0.4])
        sv = get_dd_shared_values(u_right, n_overlap=1)
        np.testing.assert_array_almost_equal(sv, [0.6, 0.8])

        # 2 domains, overplap 2
        sub_grids = make_sub_grids(grid, n_domains=2, n_overlap=2)
        u_left = sub_grids[0][1:-1] # for u(x) = x, boundary points not included.
        u_right = sub_grids[1][1:-1] # for u(x) = x, boundary points not included.
        sv = get_dd_shared_values(u_left, n_overlap=2)
        np.testing.assert_array_almost_equal(sv, [0.4, 0.3])
        sv = get_dd_shared_values(u_right, n_overlap=2)
        np.testing.assert_array_almost_equal(sv, [0.7, 0.6])

    def test_update_dd_boundary(self):

        shared_values = np.arange(1,9).reshape((4,2))
        boundary_values = np.array([
            [0, 3],
            [2, 5],
            [4, 7],
            [6, 0]])
        np.testing.assert_array_almost_equal(boundary_values, update_dd_boundary(shared_values))


class MockProblem:

    def __getattribute__(self, name):
        # The grid is not used for the test problems, so we
        # use it to return the pids of the parallel
        # subprocesses.
        if name == 'A':
            self.grid.append(os.getpid())
        return object.__getattribute__(self, name)

    def __init__(self, A, f):

        self.grid = []
        self.h = 1.
        self.boundary_values =  np.array([0,0])
        self.A = A
        self.f = f


class TestDDSolvers(unittest.TestCase):

    def setUp(self):

        self.problems = [
            MockProblem(
                A = ss.csc_matrix(np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])),
                f =  np.array([7,8,9])
            ),
            MockProblem(
                A = ss.csc_matrix(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])),
                f =  np.array([0,0,0])
            )]

        self.expected_0 = ss.linalg.spsolve(self.problems[0].A, self.problems[0].f)
        self.expected_1= np.array([self.expected_0[1], 0, 0])

    def test_solve_dd(self):

        u = solve_dd(self.problems, n_iter=2, n_overlap=1)

        problem_nr = 0
        solution = 1
        np.testing.assert_almost_equal(u[problem_nr][solution], self.expected_0)
 
        problem_nr = 1
        solution = 1
        np.testing.assert_almost_equal(u[problem_nr][solution], self.expected_1)

    def test_solve_dd_parallel(self):

        u = solve_dd_parallel(self.problems, n_iter=2, n_overlap=1)

        problem_nr = 0
        solution = 1
        np.testing.assert_almost_equal(u[problem_nr][solution], self.expected_0)

        problem_nr = 1
        solution = 1
        np.testing.assert_almost_equal(u[problem_nr][solution], self.expected_1)


        # The grid is abused to return the pids
        pids = set(sum([ui[0] for ui in u], []))
        self.assertTrue(len(pids) >= 2)


if __name__ == '__main__':

    print()
    print('Fingerprint:', hashlib.md5(open(__file__, 'rb').read()).hexdigest())
    print()
    unittest.main(verbosity=2)
