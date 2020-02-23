from numpy import meshgrid, linspace, concatenate, newaxis, empty, array, arccos, arctan, pi, sqrt
from numpy.linalg import inv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

GRID_ELEMENTS = 100
L1, L2 = 10, 10


def plane_ik(x, y):
    gamma = arccos((-L2 ** 2 + L1 ** 2 + (x ** 2 + y ** 2)) / (2 * L1 * sqrt((x ** 2 + y ** 2))))
    phi = arccos((L2 ** 2 + L1 ** 2 - (x ** 2 + y ** 2)) / (2 * L1 * L2))
    alpha = arctan(y / x)

    q1 = pi / 2 - gamma - alpha
    q2 = phi
    return array([q1, q2])


def plot_map(kc_fun):
    mesh = meshgrid(linspace(3, 10, GRID_ELEMENTS), linspace(3, 10, GRID_ELEMENTS))
    grid = concatenate([mesh[0][:, :, newaxis], mesh[1][:, :, newaxis]], axis=2)

    x_angles = empty([grid.shape[0], grid.shape[1], 2])
    y_angles = empty([grid.shape[0], grid.shape[1], 2])
    z_angles = empty([grid.shape[0], grid.shape[1], 2])

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            y_angles[i][j][:] = plane_ik(1, grid[i][j][1])
            x_angles[i][j][:] = plane_ik(grid[i][j][0], 1)
            z_angles[i][j][:] = plane_ik(grid[i][j][0], grid[i][j][1])

    deflection_x = empty([grid.shape[0], grid.shape[1]])
    deflection_y = empty([grid.shape[0], grid.shape[1]])
    deflection_z = empty([grid.shape[0], grid.shape[1]])
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            print(f'i: {str(i)}, j: {str(j)}')
            try:
                k_c_x = kc_fun('rx', *x_angles[i][j])
                k_c_y = kc_fun('ry', *y_angles[i][j])
                k_c_z = kc_fun('rz', *z_angles[i][j])
            except:
                print('oh shi')
                continue
            k_c_tot = k_c_x + k_c_y + k_c_z
            f_to_dt = inv(k_c_tot)

            deflection = f_to_dt @ array([0, 100000, 0, 0, 0, 0])

            deflection_x[i][j] = deflection[0]
            deflection_y[i][j] = deflection[1]
            deflection_z[i][j] = deflection[2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(mesh[0], mesh[1], deflection_x, alpha=0.9, cmap='Spectral')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(mesh[0], mesh[1], deflection_y, alpha=0.9, cmap='Spectral')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(mesh[0], mesh[1], deflection_z, alpha=0.9, cmap='Spectral')
    plt.show()
