from functools import reduce
from operator import mul

import sympy
from numpy import array, block, zeros, pi, eye, asarray, float64
from sympy import symbols, cos, sin, Matrix, lambdify

E = 405 * 10e9
G = 161 * 10e9
D = 0.05
k = 10e6

L1, L2 = 10, 10

translations = {'x': array([1, 0, 0]),
                'y': array([0, 1, 0]),
                'z': array([0, 0, 1])}

types = {
    # Rotation matrices
    'rx': lambda x: Matrix([[1, 0,      0,       0],
                            [0, cos(x), -sin(x), 0],
                            [0, sin(x), cos(x),  0],
                            [0, 0,      0,       1]]),

    'ry': lambda x: Matrix([[cos(x),  0, sin(x), 0],
                            [0,       1, 0,      0],
                            [-sin(x), 0, cos(x), 0],
                            [0,       0, 0,      1]]),

    'rz': lambda x: Matrix([[cos(x), -sin(x), 0, 0],
                            [sin(x), cos(x),  0, 0],
                            [0,      0,       1, 0],
                            [0,      0,       0, 1]]),
    # Translation matrices
    'tx': lambda x: Matrix([[1, 0, 0, x],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]]),

    'ty': lambda x: Matrix([[1, 0, 0, 0],
                            [0, 1, 0, x],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]]),

    'tz': lambda x: Matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, x],
                            [0, 0, 0, 1]]),
    # Rotation matrix derivatives
    'drx': lambda x: Matrix([[1, 0, 0, 0],
                             [0, -sin(x), -cos(x), 0],
                             [0, cos(x),  -sin(x), 0],
                             [0, 0,       0,       1]]),

    'dry': lambda x: Matrix([[-sin(x), 0, cos(x),  0],
                             [0,       1, 0,       0],
                             [-cos(x), 0, -sin(x), 0],
                             [0,       0, 0,       1]]),

    'drz': lambda x: Matrix([[-sin(x), -cos(x), 0, 0],
                             [cos(x),  -sin(x), 0, 0],
                             [0,       0,       1, 0],
                             [0,       0,       0, 1]]),
    # Translation matrix derivatives
    'dtx': lambda x: Matrix([[0, 0, 0, 1],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]]),

    'dty': lambda x: Matrix([[0, 0, 0, 0],
                             [0, 0, 0, 1],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]]),

    'dtz': lambda x: Matrix([[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 1],
                             [0, 0, 0, 0]])
}


class R:
    m = None
    kind = None
    parameter = None
    is_diff = True

    def __init__(self, kind=None, parameter=None, is_diff=True):
        self.is_diff = is_diff
        if not (kind is not None and parameter is not None):
            return
        self.m = types[kind](parameter)
        self.kind = kind
        self.parameter = parameter

    def derivative(self):
        if self.kind is not None and self.parameter is not None and self.is_diff:
            return R('d' + self.kind, self.parameter)
        else:
            raise AssertionError('Matrix is not differentiable')

    def __mul__(self, other):
        rotated = R(is_diff=False)
        rotated.m = self.m @ other.m
        return rotated

    def __sub__(self, other):
        rotated = R(is_diff=False)
        rotated.m = self.m - other.m
        return rotated

    def __add__(self, other):
        rotated = R(is_diff=False)
        rotated.m = self.m + other.m
        return rotated

    def __getitem__(self, item):
        return self.m[item]

    def inv(self):
        rotate = R(self.kind, None, False)
        inverse = sympy.eye(4)
        inverse[0:3, 0:3] = self.m[0:3, 0:3].transpose()
        inverse[0:3, 3] = - self.m[0:3, 0:3].transpose() * self.m[0:3, 3]
        rotate.m = inverse
        return rotate


def jac(transform):
    T_inv = reduce(mul, transform).inv()
    columns = []
    to_diff = 0
    for k in transform:
        diff_n = 0
        col = R('tx', 0)
        if not k.is_diff:
            continue
        for t in transform:
            if not t.is_diff:
                col = col * t
            else:
                if diff_n == to_diff:
                    diff_n += 1
                    col = col * t.derivative()
                else:
                    diff_n += 1
                    col = col * t
        pos = col * T_inv
        columns.append(Matrix([*col[0:3, 3], pos[2, 1], pos[0, 2], pos[1, 0]]).transpose())
        to_diff += 1
    return Matrix(columns).transpose()


def robot_stiffnesses():
    S = pi * D ** 2 / 4
    Iy = pi * D ** 4 / 64
    Iz = pi * D ** 4 / 64
    J = pi * D ** 4 / 32

    K0 = eye(6) * k
    K1 = stiffness_matrix(E, G, Iy, Iz, L1, S, J)
    K2 = stiffness_matrix(E, G, Iy, Iz, L2, S, J)
    return K0, K1, K2


def stiffness_matrix(e, g, iy, iz, l, s, j):
    return array([[e*s/l, 0,                    0,            0,         0,           0],
                  [0,     12*e*iz/l**3,         0,            0,         0,           -6*e*iz/l**2],
                  [0,     0,                    12*e*iy/l**3, 0,         6*e*iy/l**2, 0],
                  [0,     0,                    0,            g*j/l, 0,               0],
                  [0,     0,                    6*e*iy/l**2,  0,         4*e*iy/l,    0],
                  [0,     -6*e*iz/l**2,         0,            0,         0,           4*e*iz/l]])


def stiff_rot(K, axis, angle):
    r = array(R(axis, angle).m[:3, :3]).astype(float64)
    return block(
        [[r,             zeros([3, 3])],
         [zeros([3, 3]), r]]) @ K @ block(
        [[r.T,         zeros([3, 3])],
         [zeros([3, 3]), r.T]])


def get_k11(K, axis):
    r = array(R(axis, pi).m[:3, :3]).astype(float64)
    return block(
        [[r.T,         zeros([3, 3])],
         [zeros([3, 3]), r.T]]) @ K @ block(
        [[r,             zeros([3, 3])],
         [zeros([3, 3]), r]])


def skew(x):
    return array([[0,     -x[2], x[1]],
                  [x[2],  0,     -x[0]],
                  [-x[1], x[0],  0]])


# only for equal-length links
# for others replace L1 with actual length of link
def get_k12(K22, axis):
    skew_matrix = skew(translations[axis[1]]*L1)
    return - block([[eye(3),        zeros([3, 3])],
                    [skew_matrix.T, eye(3)]]) @ K22


def get_k21(K11, axis):
    skew_matrix = skew(translations[axis[1]]*L1)
    return - block([[eye(3),        zeros([3, 3])],
                    [skew_matrix,   eye(3)]]) @ K11


