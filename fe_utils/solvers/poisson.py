"""Solve a model Poisson problem with Dirichlet boundary conditions.

If run as a script, the result is plotted. This file can also be
imported as a module and convergence tests run on the solver.
"""
from fe_utils import *
import numpy as np
from numpy import sin, pi
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from argparse import ArgumentParser


def assemble(fs, f):
    """Assemble the finite element system for the Poisson problem given
    the function space in which to solve and the right hand side
    function."""

    if fs.mesh.entity_counts[-1] != f.function_space.mesh.entity_counts[-1]:
        raise ValueError("fs and f supposed to be defined on the same mesh")

    # Create an appropriate (complete) quadrature rule.
    quad = gauss_quadrature(fs.element.cell, fs.element.degree ** 2)

    # Locating the indices of boundary nodes
    fs_boundary = boundary_nodes(fs)

    # Tabulate the basis functions and their gradients at the quadrature points.
    local_phi = fs.element.tabulate(quad.points)
    local_phi_grad = fs.element.tabulate(quad.points, True)

    # Sparse matrices to hold results
    A = sp.lil_matrix((fs.node_count, fs.node_count))
    l = np.zeros(fs.node_count)

    for c in range(fs.mesh.entity_counts[-1]):
        # local_phi_grad.shape = (q, i, a) where q num of quad points, i num of basis funcs, a dimensions
        # jac.shape = (d, a) where a==d (d is just another dummy var.)
        # local_phi.shape = (q, i)

        jac = fs.mesh.jacobian(c)
        jac_det = np.abs(np.linalg.det(jac))
        jac_inv_t = np.linalg.inv(jac)

        c_nodes = f.function_space.cell_nodes[c, :]

        # RHS:
        cell_f = local_phi @ f.values[c_nodes].reshape((c_nodes.shape[0], 1))  # Contract f with basis func at quad
        cell_f_int = np.einsum("qi,q,q->i", local_phi, np.squeeze(cell_f, 1), quad.weights)  # Contract weights

        l[fs.cell_nodes[c, :]] += jac_det * cell_f_int

        # LFS:
        jac_grad_phi = np.einsum("da,qid->aiq", jac_inv_t, local_phi_grad)
        jac_grad_phi_squared = np.einsum("aiq,ajq->ijq", jac_grad_phi, jac_grad_phi)

        A[np.ix_(c_nodes, c_nodes)] += jac_det * (jac_grad_phi_squared @ quad.weights)  # Contract weights

    # Setting the boundary nodes to zero which enforces zero Dirichlet Boundary Conditions.
    l[fs_boundary] = 0
    A[fs_boundary, :] = 0
    A[fs_boundary, fs_boundary] = 1

    return A, l


def boundary_nodes(fs):
    """Find the list of boundary nodes in fs. This is a
    unit-square-specific solution. A more elegant solution would employ
    the mesh topology and numbering.
    """
    eps = 1.e-10

    f = Function(fs)

    def on_boundary(x):
        """Return 1 if on the boundary, 0. otherwise."""
        if x[0] < eps or x[0] > 1 - eps or x[1] < eps or x[1] > 1 - eps:
            return 1.
        else:
            return 0.

    f.interpolate(on_boundary)

    return np.flatnonzero(f.values)


def solve_poisson(degree, resolution, analytic=False, return_error=False):
    """Solve a model Poisson problem on a unit square mesh with
    ``resolution`` elements in each direction, using equispaced
    Lagrange elements of degree ``degree``."""

    # Set up the mesh, finite element and function space required.
    mesh = UnitSquareMesh(resolution, resolution)
    fe = LagrangeElement(mesh.cell, degree)
    fs = FunctionSpace(mesh, fe)

    # Create a function to hold the analytic solution for comparison purposes.
    analytic_answer = Function(fs)
    analytic_answer.interpolate(lambda x: sin(4 * pi * x[0]) * x[1] ** 2 * (1. - x[1]) ** 2)

    # If the analytic answer has been requested then bail out now.
    if analytic:
        return analytic_answer, 0.0

    # Create the right hand side function and populate it with the
    # correct values.
    f = Function(fs)
    f.interpolate(lambda x: (16 * pi ** 2 * (x[1] - 1) ** 2 * x[1] ** 2 - 2 * (x[1] - 1) ** 2 -
                             8 * (x[1] - 1) * x[1] - 2 * x[1] ** 2) * sin(4 * pi * x[0]))

    # Assemble the finite element system.
    A, l = assemble(fs, f)

    # Create the function to hold the solution.
    u = Function(fs)

    # Cast the matrix to a sparse format and use a sparse solver for
    # the linear system. This is vastly faster than the dense
    # alternative.
    A = sp.csr_matrix(A)
    u.values[:] = splinalg.spsolve(A, l)

    # Compute the L^2 error in the solution for testing purposes.
    error = errornorm(analytic_answer, u)

    if return_error:
        u.values -= analytic_answer.values

    # Return the solution and the error in the solution.
    return u, error


if __name__ == "__main__":
    parser = ArgumentParser(
        description="""Solve a Poisson problem on the unit square.""")
    parser.add_argument("--analytic", action="store_true",
                        help="Plot the analytic solution instead of solving the finite element problem.")
    parser.add_argument("--error", action="store_true",
                        help="Plot the error instead of the solution.")
    parser.add_argument("resolution", type=int, nargs=1,
                        help="The number of cells in each direction on the mesh.")
    parser.add_argument("degree", type=int, nargs=1,
                        help="The degree of the polynomial basis for the function space.")
    args = parser.parse_args()
    resolution = args.resolution[0]
    degree = args.degree[0]
    analytic = args.analytic
    plot_error = args.error

    u, error = solve_poisson(degree, resolution, analytic, plot_error)

    u.plot()
