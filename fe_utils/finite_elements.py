import copy
import numpy as np
from sympy import diff, Array, symbols, lambdify

from .reference_elements import ReferenceInterval, ReferenceTriangle, ReferenceCell
from .quadrature import gauss_quadrature
np.seterr(invalid='ignore', divide='ignore')


def lagrange_points(cell: ReferenceCell, degree):
    """Construct the locations of the equispaced Lagrange nodes on cell.

    :param cell: the :class:`~.reference_elements.ReferenceCell`
    :param degree: the degree of polynomials for which to construct nodes.

    :returns: a rank 2 :class:`~numpy.array` whose rows are the
        coordinates of the nodes.

    The implementation of this function is left as an :ref:`exercise
    <ex-lagrange-points>`.

    """
    if cell.dim > 2:
        raise NotImplementedError(f"{dimension}D is hard :(")
    
    dimension = cell.dim
    verticies = cell.vertices
    lpoints = copy.deepcopy(verticies)
    intervals = cell.topology[1]
    intervals = dict(sorted(intervals.items()))
    
    
    # create edge points in topological order
    for _, edge in intervals.items(): 
        x1 = verticies[int(edge[0])]
        x2 = verticies[int(edge[1])]
        
        edge_points_num = degree + 1
        edge_points = np.linspace(x1, x2, edge_points_num)[1: -1] # create p+1 equidistance points on an edge (inc. vertices)
        # print(lpoints, inner_points)
        lpoints = np.concatenate((lpoints,edge_points))
    
    # create inner points in some order
    if dimension == 2 and degree >= 3:
        for i in range(1, degree-1):
            num_interval_points = degree + 1 - i # amount of interval points is decreasing w. i
            x_i = [0, i]
            x_end = [num_interval_points - 1, i] # x-coord. of interval end-point is (deg-i, i)
            
            inner_points = 1/degree * np.linspace(x_i, x_end, num_interval_points)[1: -1]
            lpoints = np.concatenate((lpoints, inner_points))

    return lpoints
    

def _get_vandermonde_row(p, degree):
    """Internal utility function to calculate a row of the
    Vandermonde matrix for one 1D or 2D point.

    :param p: a 1D or 2D tuple of coordinates 
    :param degree: degree of polynomials 
    
    :returns: a numpy array containing the Vandermonde row for the point
    """
    
    if len(p) == 1:
        return np.vander(p, degree+1, increasing=True)[0]
    
    row = np.array([])
    for i in range(1, degree+2): # vandermonde w deg=1 has i=1, 2
        x, y = p[0], p[1]
        x_powers = np.vander([x], i)[0]
        y_powers = np.vander([y], i, increasing=True)[0]
        
        vander_row = x_powers*y_powers
        row = np.append(row, vander_row)
        
        
    return row

def _vandermonde_matrix(cell: ReferenceCell, degree: int, points):
        
    for i, x in enumerate(points):
        if not i:
            vander_row = _get_vandermonde_row(x, degree)
            vandermonde = vander_row.reshape((1, len(vander_row))) # reshape array into 1-row matrix 
            continue
        
        x_row = _get_vandermonde_row(x, degree)
        vandermonde = np.vstack((vandermonde, x_row))
    
    return vandermonde


def _vandermonde_matrix_grad(cell: ReferenceCell, degree: int, points):
    x, y = symbols("x y")
    symbol_point = [x, y] if cell.dim == 2 else [x]
    
    symbol_vander_arr = Array(_vandermonde_matrix(cell, degree, [symbol_point])[0])
    x_grad_fn = lambdify(symbol_point, symbol_vander_arr.diff(x), 'numpy')
    
    if cell.dim == 2:
        y_grad_fn = lambdify(symbol_point, symbol_vander_arr.diff(y), 'numpy')
    
    gradmonde = np.zeros((len(points), len(symbol_vander_arr), cell.dim))
    for i, p in enumerate(points):
        if cell.dim == 1:
            x_grad = x_grad_fn(p[0])
            level = np.array([x_grad])
            
        else:
            x_grad = x_grad_fn(x=p[0], y=p[1])
            y_grad = y_grad_fn(x=p[0], y=p[1])
            level = np.array([x_grad, y_grad])
            
        gradmonde[i] = np.transpose(level)
        
    return gradmonde

        

def vandermonde_matrix(cell: ReferenceCell, degree: int, points, grad=False):
    """Construct the generalised Vandermonde matrix for polynomials of the
    specified degree on the cell provided.

    :param cell: the :class:`~.reference_elements.ReferenceCell`
    :param degree: the degree of polynomials for which to construct the matrix.
    :param points: a list of coordinate tuples corresponding to the points.
    :param grad: whether to evaluate the Vandermonde matrix or its gradient.

    :returns: the generalised :ref:`Vandermonde matrix <sec-vandermonde>`

    The implementation of this function is left as an :ref:`exercise
    <ex-vandermonde>`.
    """
    if cell.dim > 2:
        raise NotImplemented(f"{cell.dim}D is hard :(")

    return _vandermonde_matrix_grad(cell, degree, points) if grad \
            else _vandermonde_matrix(cell, degree, points)


class FiniteElement(object):
    def __init__(self, cell, degree, nodes, entity_nodes=None):
        """A finite element defined over cell.

        :param cell: the :class:`~.reference_elements.ReferenceCell`
            over which the element is defined.
        :param degree: the
            polynomial degree of the element. We assume the element
            spans the complete polynomial space.
        :param nodes: a list of coordinate tuples corresponding to
            point evaluation node locations on the element.
        :param entity_nodes: a dictionary of dictionaries such that
            entity_nodes[d][i] is the list of nodes associated with
            entity `(d, i)` of dimension `d` and index `i`.

        Most of the implementation of this class is left as exercises.
        """

        #: The :class:`~.reference_elements.ReferenceCell`
        #: over which the element is defined.
        self.cell = cell
        #: The polynomial degree of the element. We assume the element
        #: spans the complete polynomial space.
        self.degree = degree
        #: The list of coordinate tuples corresponding to the nodes of
        #: the element.
        self.nodes = nodes
        #: A dictionary of dictionaries such that ``entity_nodes[d][i]``
        #: is the list of nodes associated with entity `(d, i)`.
        self.entity_nodes = entity_nodes

        if entity_nodes:
            #: ``nodes_per_entity[d]`` is the number of entities
            #: associated with an entity of dimension d.
            self.nodes_per_entity = np.array([len(entity_nodes[d][0])
                                              for d in range(cell.dim+1)])

        # Replace this exception with some code which sets
        # self.basis_coefs
        # to an array of polynomial coefficients defining the basis functions.
        self.basis_coefs = np.linalg.inv(
            vandermonde_matrix(
                    self.cell,
                    self.degree,
                    self.nodes,
                )
            )

        #: The number of nodes in this element.
        self.node_count = nodes.shape[0]

    def tabulate(self, points, grad=False):
        """Evaluate the basis functions of this finite element at the points
        provided.

        :param points: a list of coordinate tuples at which to
            tabulate the basis.
        :param grad: whether to return the tabulation of the basis or the
            tabulation of the gradient of the basis.

        :result: an array containing the value of each basis function
            at each point. If `grad` is `True`, the gradient vector of
            each basis vector at each point is returned as a rank 3
            array. The shape of the array is (points, nodes) if
            ``grad`` is ``False`` and (points, nodes, dim) if ``grad``
            is ``True``.

        The implementation of this method is left as an :ref:`exercise
        <ex-tabulate>`.

        """
        
        vander_matrix = vandermonde_matrix(self.cell, self.degree, points, grad)
        if grad:
            return np.einsum("ilk,lj -> ijk", vander_matrix, self.basis_coefs)
        
        tabulation = vander_matrix @ self.basis_coefs
        return tabulation
        
        



    def interpolate(self, fn):
        """Interpolate fn onto this finite element by evaluating it
        at each of the nodes.

        :param fn: A function ``fn(X)`` which takes a coordinate
           vector and returns a scalar value.

        :returns: A vector containing the value of ``fn`` at each node
           of this element.

        The implementation of this method is left as an :ref:`exercise
        <ex-interpolate>`.

        """

        return np.array([fn(node) for node in self.nodes])

    def __repr__(self):
        return "%s(%s, %s)" % (self.__class__.__name__,
                               self.cell,
                               self.degree)


class VectorFiniteElement(FiniteElement):
    def __init__(self, fe: FiniteElement):
        super().__init__(fe.cell, fe.degree, fe.nodes, fe.entity_nodes)

        self.d = fe.cell.dim
        self.scalar_finite_element = fe

        # Entity nodes

        for delta, d_nodes in self.entity_nodes.items():
            # e.g. delta = 1, d_nodes = {0: [0, 1], 1: [3, 4]}
            for i in d_nodes:
                if self.entity_nodes[delta][i]:
                    # Expand each node number n with the new node numberings (d*n, ..., d*n + d), then concat.
                    self.entity_nodes[delta][i] = np.concatenate([np.arange(self.d*n, self.d*(n+1)) for n in self.entity_nodes[delta][i]])
                else:
                    continue

        # On each entity we have d times number of nodes.
        self.nodes_per_entity *= self.d

        # The nodes (coordinates) are the same for nodes at the same location
        self.nodes = self.nodes.repeat(self.d, axis=0)

        # Node weights
        # self.node_weights = np.eye(d)
        self.node_weights = np.tile(np.eye(self.d), (self.node_count, 1)) # Repeat eye for each *distinct* node

        # Node count (indistinct nodes)
        self.node_count = self.nodes.shape[0]

    def tabulate(self, points, grad=False):
        scalar_tab = self.scalar_finite_element.tabulate(points, grad)

        if grad:  # TODO: verify grad implementation. Might be wrong. Maybe do the whole thing without a loop.
            res = np.zeros((self.d * scalar_tab.shape[0], scalar_tab.shape[1], scalar_tab.shape[2], self.d),
                           dtype=scalar_tab.dtype)
            for i in range(self.d):
                res[i::self.d, :, :, i] = scalar_tab

        else:
            #  e.g. d==2, then whenever i is odd we get res[i, :, 0] is zero, and for i even res[i, :, 1] is zeros.
            res = np.zeros((self.d*scalar_tab.shape[0], scalar_tab.shape[1], self.d), dtype=scalar_tab.dtype)
            for i in range(self.d):
                res[i::self.d, :, i] = scalar_tab

        return res


class LagrangeElement(FiniteElement):
    def __init__(self, cell, degree):
        """An equispaced Lagrange finite element.

        :param cell: the :class:`~.reference_elements.ReferenceCell`
            over which the element is defined.
        :param degree: the
            polynomial degree of the element. We assume the element
            spans the complete polynomial space.

        The implementation of this class is left as an :ref:`exercise
        <ex-lagrange-element>`.
        """
        if cell.dim > 2:
            raise NotImplemented(f"{cell.dim}D is hard :(")

        nodes = lagrange_points(cell, degree)  # Assuming nodes are in entity order
        n = len(nodes)
        node_indices = list(range(n))

        edge_pts_num = degree + 1 - 2
        entity_nodes = {
            0: {i: [node_indices[i]] for i in range(cell.dim + 1)}  # Vertices first
        }

        if cell.dim == 2:
            entity_nodes[1] = {
                e: node_indices[3 + e*edge_pts_num: 3 + (1 + e)*edge_pts_num] for e in range(3)
            }

            entity_nodes[2] = {0: node_indices[3 + 3*edge_pts_num:]}

        else:
            entity_nodes[1] = {0: node_indices[2:]}

        super(LagrangeElement, self).__init__(cell, degree, nodes, entity_nodes)


if __name__ == "__main__":
    lag_element = LagrangeElement(ReferenceTriangle, 3)
    vec_fe = VectorFiniteElement(lag_element)

    quad = gauss_quadrature(lag_element.cell, 3)
    tab = vec_fe.tabulate(quad.points, grad=True)
    # print(tab, tab.shape)
    # print(tab, tab.shape)
    print(tab[:, :, :, 0])
    print(tab[:, :, :, 1])



    # print(vec_fe.nodes, vec_fe.nodes_per_entity, vec_fe.node_count, vec_fe.entity_nodes, sep="\n")
    # print(len(vec_fe.node_weights), vec_fe.node_count)