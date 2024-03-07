import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation

from . import ReferenceTriangle, ReferenceInterval
from .finite_elements import FiniteElement, LagrangeElement, lagrange_points
from .mesh import Mesh, UnitSquareMesh, UnitIntervalMesh
from .quadrature import gauss_quadrature


class FunctionSpace(object):

    def __init__(self, mesh: Mesh, element: FiniteElement):
        """A finite element space.

        :param mesh: The :class:`~.mesh.Mesh` on which this space is built.
        :param element: The :class:`~.finite_elements.FiniteElement` of this
            space.

        Most of the implementation of this class is left as an :ref:`exercise
        <ex-function-space>`.
        """

        #: The :class:`~.mesh.Mesh` on which this space is built.
        self.mesh = mesh
        #: The :class:`~.finite_elements.FiniteElement` of this space.
        self.element = element

        # Implement global numbering in order to produce the global
        # cell node list for this space.
        n_cell = len(self.mesh.cell_vertices)
        max_entity_nodes = max([self.element.nodes_per_entity[d] for d in range(self.mesh.dim)])

        # Global numbering
        g = lambda d, i: (i*self.element.nodes_per_entity[d] +
                          sum([self.element.nodes_per_entity[delta] * self.mesh.entity_counts[delta]
                               for delta in range(d)]))

        dim_c = mesh.dim
        cell_nodes = np.zeros((n_cell, len(self.element.nodes)), dtype=int)
        for c in range(n_cell):
            for delta in range(dim_c + 1):
                for epsilon in range(len(self.element.entity_nodes[delta])):  # epsilon is num of nodes in entity delta
                    indices = self.element.entity_nodes[delta][epsilon]
                    i = self.mesh.adjacency(dim_c, delta)[c, epsilon] if delta != dim_c else c

                    cell_nodes[c, indices] = g(delta, i) + np.arange(self.element.nodes_per_entity[delta])

        #: The global cell node list. This is a two-dimensional array in
        #: which each row lists the global nodes incident to the corresponding
        #: cell. The implementation of this member is left as an
        #: :ref:`exercise <ex-function-space>`
        self.cell_nodes = cell_nodes

        #: The total number of nodes in the function space.
        self.node_count = np.dot(element.nodes_per_entity, mesh.entity_counts)

    def __repr__(self):
        return "%s(%s, %s)" % (self.__class__.__name__,
                               self.mesh,
                               self.element)


class Function(object):
    def __init__(self, function_space: FunctionSpace, name=None):
        """A function in a finite element space. The main role of this object
        is to store the basis function coefficients associated with the nodes
        of the underlying function space.

        :param function_space: The :class:`FunctionSpace` in which
            this :class:`Function` lives.
        :param name: An optional label for this :class:`Function`
            which will be used in output and is useful for debugging.
        """

        #: The :class:`FunctionSpace` in which this :class:`Function` lives.
        self.function_space = function_space

        #: The (optional) name of this :class:`Function`
        self.name = name

        #: The basis function coefficient values for this :class:`Function`
        self.values = np.zeros(function_space.node_count)

    def interpolate(self, fn):
        """Interpolate a given Python function onto this finite element
        :class:`Function`.

        :param fn: A function ``fn(X)`` which takes a coordinate
          vector and returns a scalar value.

        """

        fs = self.function_space

        # Create a map from the vertices to the element nodes on the
        # reference cell.
        cg1 = LagrangeElement(fs.element.cell, 1)
        coord_map = cg1.tabulate(fs.element.nodes)
        cg1fs = FunctionSpace(fs.mesh, cg1)

        for c in range(fs.mesh.entity_counts[-1]):
            # Interpolate the coordinates to the cell nodes.
            vertex_coords = fs.mesh.vertex_coords[cg1fs.cell_nodes[c, :], :]
            node_coords = np.dot(coord_map, vertex_coords)

            self.values[fs.cell_nodes[c, :]] = [fn(x) for x in node_coords]

    def plot(self, subdivisions=None):
        """Plot the value of this :class:`Function`. This is quite a low
        performance plotting routine so it will perform poorly on
        larger meshes, but it has the advantage of supporting higher
        order function spaces than many widely available libraries.

        :param subdivisions: The number of points in each direction to
          use in representing each element. The default is
          :math:`2d+1` where :math:`d` is the degree of the
          :class:`FunctionSpace`. Higher values produce prettier plots
          which render more slowly!

        """

        fs = self.function_space

        d = subdivisions or (
            2 * (fs.element.degree + 1) if fs.element.degree > 1 else 2
        )

        if fs.element.cell is ReferenceInterval:
            fig = plt.figure()
            fig.add_subplot(111)
            # Interpolation rule for element values.
            local_coords = lagrange_points(fs.element.cell, d)

        elif fs.element.cell is ReferenceTriangle:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            local_coords, triangles = self._lagrange_triangles(d)

        else:
            raise ValueError("Unknown reference cell: %s" % fs.element.cell)

        function_map = fs.element.tabulate(local_coords)

        # Interpolation rule for coordinates.
        cg1 = LagrangeElement(fs.element.cell, 1)
        coord_map = cg1.tabulate(local_coords)
        cg1fs = FunctionSpace(fs.mesh, cg1)

        for c in range(fs.mesh.entity_counts[-1]):
            vertex_coords = fs.mesh.vertex_coords[cg1fs.cell_nodes[c, :], :]
            x = np.dot(coord_map, vertex_coords)

            local_function_coefs = self.values[fs.cell_nodes[c, :]]
            v = np.dot(function_map, local_function_coefs)

            if fs.element.cell is ReferenceInterval:

                plt.plot(x[:, 0], v, 'k')

            else:
                ax.plot_trisurf(Triangulation(x[:, 0], x[:, 1], triangles),
                                v, linewidth=0)

        plt.show()

    @staticmethod
    def _lagrange_triangles(degree):
        # Triangles linking the Lagrange points.

        return (np.array([[i / degree, j / degree]
                          for j in range(degree + 1)
                          for i in range(degree + 1 - j)]),
                np.array(
                    # Up triangles
                    [np.add(np.sum(range(degree + 2 - j, degree + 2)),
                            (i, i + 1, i + degree + 1 - j))
                     for j in range(degree)
                     for i in range(degree - j)]
                    # Down triangles.
                    + [
                        np.add(
                            np.sum(range(degree + 2 - j, degree + 2)),
                            (i+1, i + degree + 1 - j + 1, i + degree + 1 - j))
                        for j in range(degree - 1)
                        for i in range(degree - 1 - j)
                    ]))

    def integrate(self):
        """Integrate this :class:`Function` over the domain.

        :result: The integral (a scalar)."""

        fs = self.function_space

        quad = gauss_quadrature(fs.element.cell, fs.element.degree)
        local_phi = fs.element.tabulate(quad.points)

        cell_integrals = np.zeros(fs.mesh.entity_counts[-1])
        for c in range(fs.mesh.entity_counts[-1]):
            func_cell_coefs = self.values[fs.cell_nodes[c, :]]
            func_cell_coefs = func_cell_coefs.reshape((func_cell_coefs.shape[0], 1))

            # local_phi @ func_cell_coefs gives a vector of sums, each component corresponds to quadrature point,
            # and the sum is over all basis functions and func coefs at that point. Then contract it with quad weights.
            cell_quad = quad.weights.reshape((1, quad.weights.shape[0])) @ (local_phi @ func_cell_coefs)
            cell_quad = cell_quad[0, 0]

            jacobian = np.abs(np.linalg.det(fs.mesh.jacobian(c)))
            cell_integrals[c] = cell_quad * jacobian

        return np.sum(cell_integrals)


if __name__ == "__main__":
    mesh = UnitIntervalMesh(20,)
    element = LagrangeElement(ReferenceInterval, 5)
    fs = FunctionSpace(mesh, element)
    sinx = Function(fs, "sinx")
    sinx.interpolate(lambda x: np.sin(2 * np.pi * x[0]))
    print(f"I'm this close to zero: {sinx.integrate()}")
