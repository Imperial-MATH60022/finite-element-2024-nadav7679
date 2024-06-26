import numpy as np
from .quadrature import gauss_quadrature
from .finite_elements import VectorFiniteElement


def errornorm(f1, f2):
    """Calculate the L^2 norm of the difference between f1 and f2.
        If both functions are VectorFiniteElements, return 2d norm
    """

    fs1 = f1.function_space
    fs2 = f2.function_space

    fe1 = fs1.element
    fe2 = fs2.element
    mesh = fs1.mesh

    # Create a quadrature rule which is exact for (f1-f2)**2.
    Q = gauss_quadrature(fe1.cell, 2*max(fe1.degree, fe2.degree))

    # Evaluate the local basis functions at the quadrature points.
    phi = fe1.tabulate(Q.points)
    psi = fe2.tabulate(Q.points)

    norm = 0.
    for c in range(mesh.entity_counts[-1]):
        # Find the appropriate global node numbers for this cell.
        nodes1 = fs1.cell_nodes[c, :]
        nodes2 = fs2.cell_nodes[c, :]

        # Compute the change of coordinates.
        J = mesh.jacobian(c)
        detJ = np.abs(np.linalg.det(J))

        # Compute the actual cell quadrature.
        if isinstance(f1.function_space.element, VectorFiniteElement) and isinstance(f2.function_space.element, VectorFiniteElement):
            c_f1 = np.einsum("i, qil->ql", f1.values[nodes1], phi)
            c_f2 = np.einsum("i, qil->ql", f2.values[nodes2], psi)
            norm += detJ * (Q.weights @ np.linalg.norm(c_f1 - c_f2, ord=2, axis=1)**2)  # Euclid 2-norm, contract weights

        else:
            norm += np.dot((np.dot(f1.values[nodes1], phi.T) -
                            np.dot(f2.values[nodes2], psi.T))**2,
                           Q.weights) * detJ

    return norm**0.5
