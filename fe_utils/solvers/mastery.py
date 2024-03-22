"""Solve a nonlinear problem using the finite element method.
If run as a script, the result is plotted. This file can also be
imported as a module and convergence tests run on the solver.
"""
from argparse import ArgumentParser
from fe_utils import *
import numpy as np
from numpy import sin, cos, pi
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg


def assemble(fs1: FunctionSpace, fs2: FunctionSpace, f: Function, mu=1):
    """Assume f is defined on fs1.
        fs1 = V,
        fs2 = Q.

    """
    n = fs1.node_count
    m = fs2.node_count

    # Solve Tu = l, where T is block matrix composed of A, B.
    l = np.zeros(n + m)
    A = sp.lil_matrix((n, n))
    B = sp.lil_matrix((m, n))
    Z = sp.lil_matrix((m, m))

    # Quadrature
    quad = gauss_quadrature(fs1.element.cell, 2 * fs1.element.degree)

    # Tabulate
    phi = fs1.element.tabulate(quad.points)
    grad_phi = fs1.element.tabulate(quad.points, grad=True)
    psi = fs2.element.tabulate(quad.points)

    for c in range(fs1.mesh.entity_counts[-1]):
        # local_phi.shape = (q, i, d) where q is num of quad points, i is nodes, d is vector dimension

        jac = fs1.mesh.jacobian(c)
        jac_det = np.abs(np.linalg.det(jac))
        jac_inv = np.linalg.inv(jac)

        fs1_c_nodes = fs1.cell_nodes[c, :]
        fs2_c_nodes = fs2.cell_nodes[c, :]

        # Compute RHS
        f_decomposed = np.einsum("k,qkl->ql", f.values[fs1_c_nodes], phi) # f evaluated at x_q, represented as a sum of basis phi_k
        f_dot_phi = np.einsum("ql,qil->qi", f_decomposed, phi)
        l[fs1_c_nodes] += jac_det * (quad.weights.transpose() @ f_dot_phi)  # Contracting quadrature rule

        # Compute LHS - A:
        local_grad_phi = np.einsum("db, qjdl ->qjbl", jac_inv, grad_phi)
        epsilon = 0.5*(local_grad_phi + local_grad_phi.transpose((0, 1, 3, 2)))
        A[np.ix_(fs1_c_nodes, fs1_c_nodes)] += mu * jac_det * np.einsum(
            "q, qibl, qjbl->ij", quad.weights, epsilon, epsilon)

        # Compute LHS - B:
        div_phi = np.trace(local_grad_phi, axis1=2, axis2=3)
        B[np.ix_(fs2_c_nodes, fs1_c_nodes)] += jac_det * np.einsum(
            "q, qi, qj -> ij", quad.weights, psi, div_phi)

    # Boundary conditions on u:
    boundary_u = boundary_nodes(fs1)
    A[boundary_u, :] = 0
    A[boundary_u, boundary_u] = 1
    l[boundary_u] = 0

    # Boundary condition on p:
    B[0, :] = 0
    Z[0, 0] = 1
    l[n] = 0  # This is the first node for fs2

    T = sp.bmat([[A, B.T], [B, Z]], format="lil")

    return T, l


def solve_mastery(resolution, analytic=False, return_error=False):
    """This function should solve the mastery problem with the given
    resolution. It should return both the solution
    :class:`~fe_utils.function_spaces.Function` and the :math:`L^2` error in
    the solution.

    If ``analytic`` is ``True`` then it should not solve the equation
    but instead return the analytic solution. If ``return_error`` is
    true then the difference between the analytic solution and the
    numerical solution should be returned in place of the solution.
    """

    mesh = UnitSquareMesh(resolution, resolution)
    fe1 = LagrangeElement(mesh.cell, 2)
    fe2 = LagrangeElement(mesh.cell, 1)
    vec_fe = VectorFiniteElement(fe1)

    fs1 = FunctionSpace(mesh, vec_fe)
    fs2 = FunctionSpace(mesh, fe2)

    real_p = Function(fs2)
    real_u = Function(fs1)
    real_u.interpolate(lambda x:
                                (
                                    2 * pi * (1 - cos(2 * pi * x[0])) * sin(2 * pi * x[1]),
                                    -2 * pi * (1 - cos(2 * pi * x[1])) * sin(2 * pi * x[0])
                                ))


    if analytic:
        return (real_u, real_p), 0.0

    force_func = Function(fs1)
    force_func.interpolate(lambda x: (
        4.0 * pi ** 3 * (1 - cos(2 * pi * x[0])) * sin(2 * pi * x[1]) - 4.0 * pi ** 3 * sin(
            2 * pi * x[1]) * cos(2 * pi * x[0]),
        -4.0 * pi ** 3 * (1 - cos(2 * pi * x[1])) * sin(2 * pi * x[0]) + 4.0 * pi ** 3 * sin(
            2 * pi * x[0]) * cos(2 * pi * x[1])
    ))


    A, l = assemble(fs1, fs2, force_func)

    A = sp.csc_matrix(A)
    luA = sp.linalg.splu(A)

    sol = luA.solve(l)

    u = Function(fs1,)
    p = Function(fs2)

    u.values[:] = sol[:fs1.node_count]
    p.values[:] = sol[fs1.node_count:]

    # Calculate error
    error = np.sqrt(errornorm(u, real_u)**2 + errornorm(p, real_p)**2)

    return (u, p), error


def boundary_nodes(fs):
    """Find the list of boundary nodes in fs, modified to work with VectorFiniteElement.
    This is a unit-square-specific solution.
    """
    eps = 1.e-10

    f = Function(fs)

    def on_boundary(x):
        """Return 1 if on the boundary, 0. otherwise."""
        if x[0] < eps or x[0] > 1 - eps or x[1] < eps or x[1] > 1 - eps:
            return (1., 1.) if isinstance(fs.element, VectorFiniteElement) else 1.
        else:
            return (0., 0.) if isinstance(fs.element, VectorFiniteElement) else 0.

    f.interpolate(on_boundary)

    return np.flatnonzero(f.values)



if __name__ == "__main__":
    # parser = ArgumentParser(
    #     description="""Solve the mastery problem.""")
    # parser.add_argument(
    #     "--analytic", action="store_true",
    #     help="Plot the analytic solution instead of solving the finite"
    #          " element problem.")
    # parser.add_argument("--error", action="store_true",
    #                     help="Plot the error instead of the solution.")
    # parser.add_argument(
    #     "resolution", type=int, nargs=1,
    #     help="The number of cells in each direction on the mesh."
    # )
    # args = parser.parse_args()
    # resolution = args.resolution[0]
    # analytic = args.analytic
    # plot_error = args.error
    #
    # u, error = solve_mastery(resolution, analytic, plot_error)

    # u.plot()

    (u,p), error = solve_mastery(10, False, False)
    print(error)
    # u.plot()
    # p.plot()

