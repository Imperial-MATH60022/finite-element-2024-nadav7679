'''Test that an integral over VectorFiniteElement works.'''
import pytest
import numpy as np
import scipy.sparse as sp

from fe_utils import *


def assemble(fs, f):
    quad = gauss_quadrature(fs.element.cell, fs.element.degree ** 2)

    # Tabulate the basis functions and their gradients at the quadrature points.
    local_phi = fs.element.tabulate(quad.points)

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

        # RHS
        f_decomposed = np.einsum("k,qkl->ql", f.values[c_nodes],
                                 local_phi)  # f evaluated at x_q, represented as a sum of basis phi_k
        f_dot_phi = np.einsum("ql,qil->qi", f_decomposed, local_phi)
        l[c_nodes] += jac_det * np.einsum("q,qi->i", quad.weights, f_dot_phi)

        # LHS
        phi_dot_phi = np.einsum("qil, qjl->qij", local_phi, local_phi)
        A[np.ix_(c_nodes, c_nodes)] += jac_det * np.einsum("q, qij ->ij", quad.weights, phi_dot_phi)

    return A, l

def test_gradient():
    mesh = UnitSquareMesh(10, 10)
    fe = VectorFiniteElement(LagrangeElement(ReferenceTriangle, 5))
    fs = FunctionSpace(mesh, fe)

    grad_f = Function(fs)
    grad_f.interpolate(lambda x: (2 * x[0], 2 * x[1]))  # Gradiant of (x**2 + y**2), supposedly.

    A, l = assemble(fs, grad_f)

    u = Function(fs)

    A = sp.csr_matrix(A)
    u.values[:] = sp.linalg.spsolve(A, l)

    err = Function(fs)
    err.values = u.values - grad_f.values
    assert all(err.values < 10E-08)

    # print(err.values)
    # print(errornorm(err, err))
    # err.plot()



if __name__ == '__main__':
    import sys

    pytest.main(sys.argv)
    # test_gradient()
