import numpy
from numpy import array, block, arctan, arccos, sqrt, pi, zeros
from numpy.linalg import inv
from sympy import symbols, eye, lambdify
from sympy.abc import x, y, z

from Helpers.helpers import R, jac, robot_stiffnesses

GRID_ELEMENTS = 20

L1, L2 = 10, 10

m = eye(5)


def two_jacs(axis):
    r, t, ax = axis
    q1, q2 = symbols('q1 q2')
    t2 = [R(r, ax, is_diff=False),
          R('rx', 0), R('ry', 0), R('rz', 0), R('tx', 0), R('ty', 0), R('tz', 0),
          R(r, q1, is_diff=False), R(t, L1, is_diff=False),
          R('rx', 0), R('ry', 0), R('rz', 0), R('tx', 0), R('ty', 0), R('tz', 0),
          R(r, q2, is_diff=False),
          R('rx', 0), R('ry', 0), R('rz', 0), R('tx', 0), R('ty', 0), R('tz', 0)
          ]

    t1 = [R(r, ax, is_diff=False),
          R('rx', 0, is_diff=False), R('ry', 0, is_diff=False), R('rz', 0, is_diff=False),
          R(r, q1), R(t, L1, is_diff=False),
          R('rx', 0, is_diff=False), R('ry', 0, is_diff=False), R('rz', 0, is_diff=False),
          R('tx', 0, is_diff=False), R('ty', 0, is_diff=False), R('tz', 0, is_diff=False),
          R(r, q2),
          R('rx', 0, is_diff=False), R('ry', 0, is_diff=False), R('rz', 0, is_diff=False),
          R('tx', 0, is_diff=False), R('ty', 0, is_diff=False), R('tz', 0, is_diff=False)
          ]

    j_t = jac(t2)
    j_t = lambdify([ax, q1, q2], j_t, 'numpy')
    j_q = jac(t1)
    j_q = lambdify([ax, q1, q2], j_q, 'numpy')
    return j_t, j_q


def plane_ik(x, y):
    gamma = arccos((-L2**2 + L1**2 + (x**2 + y**2)) / (2 * L1 * sqrt((x**2 + y**2))))
    phi = arccos((L2**2 + L1**2 - (x**2 + y**2)) / (2 * L1 * L2))
    alpha = arctan(y / x)

    q1 = pi / 2 - gamma - alpha
    q2 = phi
    return array([q1, q2])


def get_kc_fsa(axis, q_1, q_2):
    jt, jq = axis_selector[axis]
    j_t = jt(1, q_1, q_2)
    j_q = jq(1, q_1, q_2)
    mat = block([[j_t @ K_theta_inv @ j_t.T, j_q],
                 [j_q.T, numpy.zeros((2, 2))]])
    k_c = inv(mat)[:6, :6]
    return k_c


# Lets made out of wolfram in form of cylindrical rod
# Same for all links

K0, K1, K2 = robot_stiffnesses()

K_theta_inv = inv(block([[K0,            zeros([6, 6]), zeros([6, 6])],
                         [zeros([6, 6]), K1,            zeros([6, 6])],
                         [zeros([6, 6]), zeros([6, 6]), K2]]))

j_theta_x, jq_x = two_jacs(['rx', 'tz', x])
j_theta_y, jq_y = two_jacs(['ry', 'tx', y])
j_theta_z, jq_z = two_jacs(['rz', 'ty', z])

axis_selector = {'rx': two_jacs(['rx', 'tz', x]),
                 'ry': two_jacs(['ry', 'tx', x]),
                 'rz': two_jacs(['rz', 'ty', x])}
