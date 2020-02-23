from functools import reduce
from operator import mul

import numpy
from numpy import arctan2, sqrt, pi, sin, cos, array, zeros, block, asarray, newaxis, hstack, vstack
from numpy.linalg import pinv, inv
from scipy.stats import truncnorm, randint, uniform

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy import symbols, lambdify, Matrix, BlockMatrix

from Helpers.helpers import R, jac

# Sizes in feet
# Stiffness in lbf/ft

L1 = 1
MAX_Q2 = 5

K1 = 73756.215
K2 = 147512.429
K3 = 368781.075

L2 = 2
MAX_Q3 = 3

MEAN_Z = 3.5
MEAN_R = 7
STD = 1

q1, q2, q3, t1, t2, t3 = symbols('q1, q2, q3, t1, t2, t3')
fx, fy, fz, rx, ry, rz = symbols('fx, fy, fz, rx, ry, rz')

model = [R('rz', -q1, is_diff=False), R('rz', t1), R('tz', L1, is_diff=False),
         R('tz', q2, is_diff=False), R('tz', t2),
         R('ty', L2, is_diff=False), R('ty', q3, is_diff=False), R('ty', t3)]
fk = lambdify([q1, q2, q3, t1, t2, t3], reduce(mul, model).m, 'numpy')


def ik(x, y, z):
    return arctan2(x, y), z - L1, sqrt(x ** 2 + y ** 2) - L2


def gen_points(n):
    z = -truncnorm.rvs((L1-MEAN_Z)/STD, (L1+MAX_Q2-MEAN_Z)/STD, loc=MEAN_Z, scale=STD, size=n)
    angle = uniform.rvs(0, 2*pi, n)
    r = truncnorm.rvs((L2-MEAN_R)/STD, (L1+MAX_Q3-MEAN_R)/STD, loc=MEAN_R, scale=4*STD, size=n)
    x, y = sin(angle) * r, cos(angle) * r
    return asarray([x, y, z], dtype=numpy.float64)


def deflect(coord, force):
    angles = ik(*coord)
    dq1, dq3, dq2 = force[:3] / array([K1, K2, K3]) * array([L2 + angles[2], 1, 1])
    loaded = fk(*angles, dq1, dq2, dq3)[:3, 3]
    return [dq1, dq2, dq3, 0, 0, dq1], loaded


a = gen_points(3000)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(*a)
ax.set_xlabel('x, feet')
ax.set_ylabel('y, feet')
ax.set_zlabel('z, feet')
#plt.show()

j = jac(model)

A = []
W = array([500, 500, 500, 0, 0, 0])
for x in range(j.shape[1]):
    A.append(j[:, x] * j[:, x].T * Matrix([fx, fy, fz, rx, ry, rz]))
A = BlockMatrix(A)
A = lambdify([q1, q2, q3, t1, t2, t3, fx, fy, fz, rx, ry, rz], A, 'numpy')

A_A_T = None
A_T = None
for x in range(len(a[0])):
    deflections, coords = deflect(a[:, x], W)
    At = A(*coords, *deflections[:3], *W)
    A_A_T = A_A_T + At.T @ At if A_A_T is not None else At.T @ At
    A_T = A_T + At.T @ deflections if A_T is not None else At.T @ deflections

print(inv(A_A_T) @ A_T)

