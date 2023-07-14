import numpy as np
import matplotlib.pyplot as plt

from matplotlib.tri import Triangulation


class Node:
    def __init__(self, x, y, i, j, index, is_domain):
        self.x = x
        self.y = y
        self.i = i
        self.j = j
        self.is_domain = is_domain
        self.index = index if self.is_domain else None

        self.V = None
        self.sigma = None

class Triangle:
    def __init__(self, nodes, centroid, index, in_domain):
        self.nodes = nodes
        self.centroid = centroid
        self.area = self._calc_area()
        self.in_domain = in_domain
        self.index = index if self.in_domain else None
        self.sigma = None

    def _calc_area(self):
        x1, y1 = self.nodes[0].x, self.nodes[0].y
        x2, y2 = self.nodes[1].x, self.nodes[1].y
        x3, y3 = self.nodes[2].x, self.nodes[2].y
        return abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)) * 0.5

    def calc_matrix(self):
        x1, y1 = self.nodes[0].x, self.nodes[0].y
        x2, y2 = self.nodes[1].x, self.nodes[1].y
        x3, y3 = self.nodes[2].x, self.nodes[2].y

        self.grad = np.array([
            [y3-y2, x2-x3],
            [y1-y3, x3-x1],
            [y2-y1, x1-x2],
        ]) / (2*self.area)

        # Compute the element matrix
        self.K = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                self.K[i, j] = self.sigma * self.area * np.dot(self.grad[i, :], self.grad[j, :])

    def calc_current_density(self):
        # Calculate the gradient of the potential
        grad_V = np.zeros(2)
        for i, node in enumerate(self.nodes):
            grad_V += node.V * self.grad[i, :]

        # Calculate the current density
        self.J = -self.sigma * grad_V
        self.J_mag = np.sqrt(self.J[0]**2 + self.J[1]**2)

class Mesh:
    def __init__(self, total_width, total_height, thickness, depth, step):
        self.total_width = total_width
        self.total_height = total_height
        self.thickness = thickness
        self.depth = depth
        self.step = step

        self.n_rows = int(self.total_height / self.step) + 1
        self.n_cols = int(self.total_width / self.step) + 1

        self.nodes = self._create_domain()
        self.elements = self._create_elements()
        self._assign_materials()

    def _create_matrix(self, element=None):
        return np.array([[element] * self.n_cols] * self.n_rows)

    def _create_domain(self):
        nodes = self._create_matrix()
        counter = 0

        for i in range(self.n_rows):
            for j in range(self.n_cols):
                x = j * self.step
                y = i * self.step

                is_domain = (x < self.thickness or y < self.thickness) or np.isclose(x, self.thickness) or np.isclose(y, self.thickness)

                if is_domain: counter += 1
                nodes[i, j] = Node(x, y, i, j, counter, is_domain)
        return nodes

    def _create_squared_mesh(self):



        squares = []
        for i in range(self.n_rows - 1):
            for j in range(self.n_cols - 1):
                square = [
                    self.nodes[i, j],
                    self.nodes[i+1, j],
                    self.nodes[i+1, j+1],
                    self.nodes[i, j+1]
                ]
                squares.append(square)
        return squares

    def _split_squares(self, squares):
        def calc_centroid(nodes):
            return np.mean([(node.x, node.y) for node in nodes], axis=0)

        elements = []
        counter = 0
        for square in squares:
            lower_triangle = [None, None, None]
            upper_triangle = [None, None, None]

            # Counter clockwise
            lower_triangle[0] = square[0]
            lower_triangle[1] = square[1]
            lower_triangle[2] = square[3]

            # Counter clockwise
            upper_triangle[0] = square[1]
            upper_triangle[1] = square[2]
            upper_triangle[2] = square[3]

            for nodes in [lower_triangle, upper_triangle]:
                centroid = calc_centroid(nodes)
                in_domain = centroid[0] <= self.thickness or centroid[1] <= self.thickness
                if in_domain: counter += 1
                elements.append(Triangle(nodes, centroid, counter, in_domain))
        return elements

    def _create_elements(self):
        return self._split_squares(self._create_squared_mesh())

    def _assign_materials(self):
        sigma_a = 4e6
        sigma_b = 9e7

        for triangle in self.elements:
            if (0.03 < triangle.centroid[0] < 0.06) and (0.03 < triangle.centroid[1] < 0.15):
                triangle.sigma = sigma_b
                triangle.calc_matrix()
            elif (0.06 < triangle.centroid[0] < 0.09) and (0.03 < triangle.centroid[1] < 0.06):
                triangle.sigma = sigma_b
                triangle.calc_matrix()
            elif (0.09 < triangle.centroid[0] < 0.18) and (0.00 < triangle.centroid[1] < 0.09):
                triangle.sigma = sigma_b
                triangle.calc_matrix()
            else:
                triangle.sigma = sigma_a
                triangle.calc_matrix()

    def assemble(self):
        n = np.sum([node.is_domain for row in self.nodes for node in row])

        # Initialize the global matrix
        self.K = np.zeros((n, n))

        for element in self.elements:
            if element.in_domain:
                # Get the global indices of the nodes
                indices = [node.index-1 for node in element.nodes]  # Minus 1 because Python indices start at 0

                # Add the element matrix to the global matrix
                for a in range(3):
                    for b in range(3):
                        self.K[indices[a], indices[b]] += element.K[a, b]

    def apply_boundary(self):
        # Initialize the rhs vector
        self.F = np.zeros(self.K.shape[0])

        for node in np.ravel(self.nodes):
            if node.is_domain:
                # Get the global index of the node
                idx = node.index - 1

                # Check if the node is on the left or top boundary
                if np.isclose(node.x, 0) or np.isclose(node.y, self.total_height):
                    self.K[idx, :] = 0
                    self.K[idx, idx] = 1
                    self.F[idx] = 200  # Prescribed boundary value

                # Check if the node is on the right boundary
                elif np.isclose(node.x, self.total_width):
                    self.K[idx, :] = 0
                    self.K[idx, idx] = 1
                    self.F[idx] = 0  # Prescribed boundary value

    def solve(self):
        self.assemble()
        self.apply_boundary()

        self.assign_solution(np.linalg.solve(self.K, self.F))

    def plot_grid(self):
        for triangle in self.elements:
            if not triangle.in_domain: continue

            x_vals = [node.x for node in triangle.nodes] + [triangle.nodes[0].x]
            y_vals = [node.y for node in triangle.nodes] + [triangle.nodes[0].y]

            color = 'green' if triangle.sigma == 9e7 else 'yellow'

            plt.plot(x_vals, y_vals, c='k', linewidth=1)
            plt.fill_between(x_vals, y_vals, color=color, alpha=0.9)

            # Display element index at centroid of the triangle
            plt.text(triangle.centroid[0], triangle.centroid[1], str(triangle.index), fontsize=8, c='k', ha='center', va='center')

        for row in self.nodes:
            for node in row:
                if node.is_domain:
                    plt.plot(node.x, node.y, 'ko', markersize=8)
                    plt.text(node.x, node.y, str(node.index), fontsize=6, c='w', ha='center', va='center')

        plt.axis('equal')
        plt.grid(True)
        plt.xticks(np.arange(0, self.total_width + self.step, self.step))
        plt.yticks(np.arange(0, self.total_height + self.step, self.step))
        plt.title('Mesh')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')

        plt.savefig('./images/mesh.svg', format='svg')
        plt.show()

    def assign_solution(self, V):
        for node in np.ravel(self.nodes):
            if node.is_domain:
                # Assign the corresponding entry of the solution to the node
                node.V = V[node.index - 1]

    def plot_solution(self):
        nodes = []
        x = []
        y = []
        V = []
        triangles = []

        for row in self.nodes:
            for node in row:
                if node.is_domain:
                    nodes.append(node)
                    x.append(node.x)
                    y.append(node.y)
                    V.append(node.V)

        for element in self.elements:
            if element.in_domain:
                # Find the indices of the nodes in the node list
                indices = [nodes.index(node) for node in element.nodes]
                triangles.append(indices)

        x = np.array(x)
        y = np.array(y)
        V = np.array(V)
        triangles = np.array(triangles)

        triangulation = Triangulation(x, y, triangles)

        # Create a filled contour plot of the triangulation
        plt.tripcolor(triangulation, V, shading='flat', cmap='viridis')
        plt.colorbar(label='Tensão (V)')
        plt.axis('equal')
        plt.grid(True)
        plt.title('Distribuição de Potencial')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')

        plt.savefig('./images/V.svg', format='svg')
        plt.show()

    def plot_current_density(self):
        # Compute the current density for each element
        for element in self.elements:
            if element.in_domain:
                element.calc_current_density()

        # Create arrays for the element centroids and current densities
        x = [element.centroid[0] for element in self.elements if element.in_domain]
        y = [element.centroid[1] for element in self.elements if element.in_domain]
        Jx = [element.J[0] for element in self.elements if element.in_domain]
        Jy = [element.J[1] for element in self.elements if element.in_domain]
        m = [element.J_mag for element in self.elements if element.in_domain]

        # Create a quiver plot of the current density
        plt.quiver(x, y, Jx, Jy, m)
        plt.axis('equal')
        plt.grid(True)
        plt.title('Densidade de Corrente')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')

        plt.savefig('./images/J.svg', format='svg')
        plt.show()

    def calc_current_across_surface(self, surface_x):
        total_current = 0.0

        for element in mesh.elements:
            if not element.in_domain: continue

            # Check if the element intersects the surface
            # If so, compute the element contribution to the total current
            x_coords = [node.x for node in element.nodes if np.isclose(node.x, surface_x)]
            if len(x_coords) == 2: total_current += element.J[0] * self.step # Trapezoid integral
        return total_current * self.depth


total_width = 0.21
total_height = 0.18
thickness = 0.09
depth = 0.2
step = 0.005

mesh = Mesh(total_width, total_height, thickness, depth, step)
mesh.plot_grid()
mesh.solve()
mesh.plot_solution()
mesh.plot_current_density()

potential_diff = 200
i_m = mesh.calc_current_across_surface(total_width)

print(f"Corrente total através do bloco: {i_m / 1e6:.4f} MA")
print(f"Resistência do bloco: {potential_diff / i_m * 1e9:.4f} nΩ")
