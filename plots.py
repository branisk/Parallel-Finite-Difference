from domain_decomposition import *
import matplotlib.pyplot as plt
plt.style.use('bmh')

def f(x):
    return 1

n_domains = 4
n_overlap = 4
n_iter = 10
cells = 20

a = 0
b = 1

grid = np.linspace(a, b, n_domains*cells + 1)
sub_grids = make_sub_grids(grid, n_domains, n_overlap)

h = 1 / (len(grid) - 1)
rhs = f

problems = []

for i in range(n_domains):
    problems.append(Poisson(sub_grids[i], h, rhs))

fig, ax = plt.subplots(1,2)

local_res = solve_dd(problems, n_iter, n_overlap)
parallel_res = solve_dd_parallel(problems, 1, n_overlap)

for i in range(4):
    ax[0].plot(local_res[i][0][1:-1], local_res[i][1])
    ax[1].plot(parallel_res[i][0][1:-1], parallel_res[i][1])

ax[0].set_xlabel("Grid points (Local)")
ax[1].set_xlabel("Grid points (Parallel)")
ax[0].set_ylabel("Solution")
ax[1].tick_params(left=False, labelleft=False)

fig.suptitle("Domain Decomposition of -u''(x)=f, u(0)=u(1)=0, f(x)=1")
plt.show()
