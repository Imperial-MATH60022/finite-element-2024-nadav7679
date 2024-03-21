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


    # Quadrature for RHS
    quad = gauss_quadrature(fs1.element.cell, fs1.element.degree ** 2)

    # Tabulate
    phi = fs1.element.tabulate(quad.points)
    grad_phi = fs1.element.tabulate(quad.points, grad=True)
    psi = fs2.element.tabulate(quad.points)

    F = np.zeros(n)
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
        F[fs1_c_nodes] += jac_det * (quad.weights.transpose() @ f_dot_phi)  # Contracting quadrature rule

        # Compute LHS - A:
        # local_grad_phi = np.einsum("db, qjad->qjab", jac_inv, grad_phi) # d for dummy
        local_grad_phi = grad_phi @ jac_inv
        epsilon_phi = 0.5*(local_grad_phi + local_grad_phi.transpose((0, 1, 3, 2)))
        epsilon_dot_epsilon = np.einsum("qiab,qjab->qij", epsilon_phi, epsilon_phi) # dot(eps(phi_i(xq)), eps(phi_j(xq))) contracted on q
        A[np.ix_(fs1_c_nodes, fs1_c_nodes)] += mu * jac_det * np.einsum("q, qij->ij", quad.weights, epsilon_dot_epsilon)


        # Compute LHS - B:
        # div_phi = np.trace(local_grad_phi, axis1=2, axis2=3)
        div_phi = np.einsum("qjaa->qj", local_grad_phi)
        B[np.ix_(fs2_c_nodes, fs1_c_nodes)] += jac_det * np.einsum("q, qi, qj -> ij", quad.weights, psi, div_phi)

    l[:n] = F
    T = sp.bmat([[A, B.transpose()], [B, None]], "lil")

    # Boundary conditions on u:
    u_boundary = boundary_nodes(fs1)
    l[u_boundary] = 0.
    T[u_boundary, :] = 1E-17  # Non convergence with 0 then roundoff error instead
    T[u_boundary, u_boundary] = 1.

    # Boundary condition on p:
    arbitrary_dof = n + 1   # 1 is arbitrary here, n is just where the enumeration begins for fs2
    l[arbitrary_dof] = 0.
    T[arbitrary_dof, :] = 1e-17
    T[arbitrary_dof, arbitrary_dof] = 1.

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

    degree = 5

    mesh = UnitSquareMesh(resolution, resolution)
    fe = LagrangeElement(mesh.cell, degree)

    vec_fe = VectorFiniteElement(fe)
    fs1 = FunctionSpace(mesh, vec_fe)

    fs2 = FunctionSpace(mesh, fe)

    # Need to create u from gamma
    analytic_answer = Function(fs1)
    analytic_answer.interpolate(lambda x:
                                (
                                    2 * pi * (1 - cos(2 * pi * x[0]) * sin(2 * pi * x[1])),
                                    2 * pi * (1 - cos(2 * pi * x[1]) * sin(2 * pi * x[0]))
                                ))

    # TODO: check that the analytic answer for u is correct. Unsure about p=0, it might be wrong.
    #       Also, need to derive f, I'm unsure about that. If p is a constant, then f=grad**2 u.
    if analytic:
        return (analytic_answer, 0), 0.0

    mu = 0.1
    force_func = Function(fs1)
    force_func.interpolate(lambda x:
                           (
                               -32 * pi * mu * cos(2 * pi * x[0]) * sin(2 * pi * x[1]),
                               32 * pi * mu * cos(2 * pi * x[1]) * sin(2 * pi * x[0]),
                           )
                           )

    A, l = assemble(fs1, fs2, force_func)

    A = sp.csc_matrix(A)
    luA = sp.linalg.splu(A)

    sol = luA.solve(l)

    u = Function(fs1,)
    p = Function(fs2)

    u.values = sol[:fs1.node_count]
    p.values = sol[:fs2.node_count]

    # Calculate error
    real_p = Function(fs2)
    error_p = errornorm(p, real_p)

    real_u = Function(fs1)
    real_u.interpolate(lambda x:
                       (
                           2*pi*(1-cos(2*pi*x[0])) * sin(2*pi*x[1]),
                           -2*pi*(1-cos(2*pi*x[1])) * sin(2*pi*x[0]),
                       )
                       )
    error_u = errornorm(u, real_u)

    error = np.sqrt(error_u**2 + error_p**2)

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
    #
    # u.plot()

    (u,p), error = solve_mastery(4, False, False)
    print(error)
    u.plot()
    p.plot()

