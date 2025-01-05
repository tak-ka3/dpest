import math
import numpy as np
from dpest.distrib import Uni
from dpest.operation import Br
from dpest.input import InputScalarToArray
from dpest.func import eps_est

eps = 0.1
n = 5
k = math.ceil(math.log(2.0/eps))
d = int((math.pow(2, k + 1) + 1)*math.pow((math.pow(2, k) + 1), n - 1))

def compute_f(c: int):
    """
    Computes the function F as defined in Step 2 of Alg. 4.8.

    Returns:
        1d-array f of shape (n+1,), where f[z] = F(z) for z = 0, 1, ..., n
    """

    z_le_c = np.atleast_2d(range(0, c))
    z_geq_c = np.atleast_2d(range(c, n))

    # compute the F function
    f = np.empty(n + 1, dtype=np.int64)

    # for interval [0, c)
    a = np.power(2, k * (c - z_le_c))
    b = np.power((np.power(2, k) + 1), n - (c - z_le_c))
    f[:c] = np.multiply(a, b)

    # for interval [c, n)
    a = np.power(2, k * (z_geq_c - c + 1))
    b = np.power((np.power(2, k) + 1), n - 1 - (z_geq_c - c))
    f[c:n] = d - np.multiply(a, b)

    # for n
    f[-1] = d

    return f

c = 1

Arr = InputScalarToArray(size=n+1, func=compute_f)
u = Uni(1, d+1)
z = 0

# Y = Br(u, Arr[0], 0, Br(u, Arr[1], 1, Br(u, Arr[2], 2, Br(u, Arr[3], 3, Br(u, Arr[4], 4, Br(u, Arr[5], 5, z))))))
for idx in reversed(range(n+1)):
    z = Br(u, Arr[idx], z, idx)
Y = z
eps = eps_est(Y)
