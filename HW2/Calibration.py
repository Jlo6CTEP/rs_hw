from functools import reduce
from operator import mul

import numpy
from numpy import arctan2, sqrt, pi, sin, cos, array, zeros, block, asarray, newaxis, hstack, vstack, full, linspace
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

K1 = 1 / 73756.215
K2 = 1 / 147512.429
K3 = 1 / 36781.075

K_REAL = array([K1, K2, K3])

L2 = 2
MAX_Q3 = 3

MEAN_Z = 3.5
MEAN_R = 7
STD = 1

q1, q2, q3 = symbols('q1, q2, q3')
fx, fy, fz, rx, ry, rz = symbols('fx, fy, fz, rx, ry, rz')

model = [R('rz', -q1, is_diff=False), R('rz', 0), R('tz', L1, is_diff=False),
         R('tz', q2, is_diff=False), R('tz', 0),
         R('ty', L2, is_diff=False), R('ty', q3, is_diff=False), R('ty', 0)]
fk = lambdify([q1, q2, q3], reduce(mul, model).m, 'numpy')


def ik(x, y, z):
    return arctan2(x, y), z - L1, sqrt(x ** 2 + y ** 2) - L2


def gen_points(n):
    z = -truncnorm.rvs((L1-MEAN_Z)/STD, (L1+MAX_Q2-MEAN_Z)/STD, loc=MEAN_Z, scale=STD, size=n)
    angle = uniform.rvs(0, 2*pi, n)
    r = truncnorm.rvs((L2-MEAN_R)/STD, (L1+MAX_Q3-MEAN_R)/STD, loc=MEAN_R, scale=4*STD, size=n)
    x, y = sin(angle) * r, cos(angle) * r
    return asarray([x, y, z], dtype=numpy.float64)


def gen_circle(n):
    z = full([n], L1 + MAX_Q2/2)
    angle = linspace(0, 2*pi, n)
    r = L2 + MAX_Q3/2
    x, y = sin(angle) * r, cos(angle) * r
    return asarray([x, y, z, zeros(n), zeros(n), angle], dtype=numpy.float64).T


a = gen_points(300)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(*a)
ax.set_xlabel('x, feet')
ax.set_ylabel('y, feet')
ax.set_zlabel('z, feet')
ax.legend()
plt.show()

j = jac(model)

A = []
W = array([500, 500, 500, 500, 500, 500])
for x in range(j.shape[1]):
    A.append(j[:, x] * j[:, x].T * Matrix([fx, fy, fz, rx, ry, rz]))
A = BlockMatrix(A)
A = lambdify([q1, q2, q3, fx, fy, fz, rx, ry, rz], A, 'numpy')

A_A_T = None
A_T = None
for x in range(len(a[0])):
    At = A(*a[:, x], *W)
    A_A_T = A_A_T + At.T @ At if A_A_T is not None else At.T @ At
    A_T = A_T + At.T @ (At @ K_REAL) if A_T is not None else At.T @ (At @ K_REAL)

k = inv(A_A_T) @ A_T

non_cal = A(1, 1, 1, *W) @ K_REAL
cal = fk(1, 1, 1)

circle = gen_circle(100)
obtained = []
calibrated = []

for x in circle:
    dx = A(*x[:3], *W) @ K_REAL
    obtained.append(dx + x)
    calibrated.append(A(*(x - dx)[:3], *W) @ k + (x - dx))

obtained = vstack(obtained).T
calibrated = vstack(calibrated).T

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(*circle.T[:3], c="blue", label="desired")
ax.plot(*obtained[:3], c="red", label="obtained")
ax.plot(*calibrated[:3], c="green", label="calibrated")
ax.set_xlabel('x, feet')
ax.set_ylabel('y, feet')
ax.set_zlabel('z, feet')
ax.set_zlim(2, 5)
ax.legend()
plt.show()
print("Given", 1/K_REAL)
print("Estimated", 1/k)

