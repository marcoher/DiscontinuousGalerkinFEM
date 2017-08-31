import numpy as np
from scipy.special import gamma

class Jacobi(object):
  """ Class for normalized Jacobi polynomials """
  def __init__(self, n, alpha, beta):
    """ Finds (d/dx)^j P_i^(alpha, beta) (x)
        i = 0 ... n, j = 0 ... m
        shape = (n+1, m+1, n+1)
        where x are the Gauss-Lobatto nodes. """
    self.n = n

    self.alpha = alpha

    self.beta = beta

    self.nodes, self.weights = self.gauss_lobatto_nodes(n, alpha, beta)

    self.P = self.eval_P_tensor(n, 0, alpha, beta, self.nodes)

  def recursion_coefs(self, n, alpha, beta):
    """ Coefficients in the three-term recursion relation"""
    nnab = 2*n + alpha + beta

    assert np.all(nnab>-1)

    a = (nnab + 1.0)*(nnab + 2.0)\
        /(2*(n + 1)*(n + alpha + beta + 1))

    b = np.where(nnab>0, (beta**2 - alpha**2)*(nnab + 1.0)\
                         /(2*(n + 1.0)*(nnab + 1.0)*nnab),
                         0.5*(beta - alpha))

    c = np.where(nnab>0, (n + alpha)*(n + beta)*(nnab + 2.0)\
                         /((n + 1)*(n + alpha + beta + 1)*nnab),
                         0)

    return a, b, c

  def norm(self, n, alpha, beta):
    """ Norm of the Jacobi polynomials"""
    nab = n + alpha + beta

    assert np.all(nab>-2)

    norm2 = np.where(nab>-1,
                     (2**(alpha + beta + 1.0) / (n + nab + 1.0))\
                       * (gamma(n + alpha +1) * gamma(n + beta + 1))\
                       / (gamma(n + 1.0) * gamma(nab + 1)),
                     2**(alpha + beta + 1.0)\
                       * (gamma(n + alpha +1) * gamma(n + beta + 1))\
                       / (gamma(n + 1.0) * gamma(nab + 2)))

    return np.sqrt(norm2)

  def gauss_nodes(self, n, alpha, beta):
    """ Jacobi-Gauss quadrature nodes and weights """
    if n==0:
      x = np.array([(alpha-beta)/(alpha + beta + 2)])

      w = np.array([2.0])

      return x, w

    elif n<0:
      x = np.array([])

      w = np.array([])

      return x, w

    k = np.arange(n+1)

    a, b, c = self.recursion_coefs(k, alpha, beta)

    d0 = b/a

    d1 = np.sqrt( c[1:]/(a[:-1]*a[1:]) )

    A = np.diag(d0) + np.diag(d1,1) + np.diag(d1,-1)

    x, v = np.linalg.eig(A)

    idx = np.argsort(x)

    x, v = x[idx], v[:,idx]

    w = self.norm(0, alpha, beta)**2 * v[0,:]**2

    return x, w

  def gauss_lobatto_nodes(self, n, alpha, beta):
    """ Jacobi-Gauss-Lobatto quadrature nodes and weights """
    x, w = self.gauss_nodes(n-2, alpha+1, beta+1)

    w = w/(1-x**2)

    w0 = 2**(alpha + beta + 1) * (gamma(n + alpha + 1)/gamma(n + beta + 1))\
         * ((beta + 1)*gamma(beta + 1)**2)\
         * (gamma(n)/gamma(n + alpha + beta + 2))

    wn = 2**(alpha + beta + 1) * (gamma(n + beta + 1)/gamma(n + alpha + 1))\
         * ((alpha + 1)*gamma(alpha + 1)**2)\
         * (gamma(n)/gamma(n + alpha + beta + 2))

    x = np.concatenate([[-1.0], x, [1.0]])

    w = np.concatenate([[w0], w, [wn]])

    return x, w

  def eval_P_tensor(self, n, m, alpha, beta, x):
    P = np.zeros(shape=(n+1,m+1,) + np.shape(x))

    nrm = self.norm(np.arange(n+1), alpha, beta).reshape((-1,1,1))

    for j in range(m+1):
      if j==0:
        P[0,j] = 1.0

        if n>0:
          P[1,j] = 0.5*((alpha + beta + 2)*x - (beta - alpha))

        if n>1:
          for k in range(1, n):
            a, b, c = self.recursion_coefs(k, alpha, beta)

            P[k+1,j] = (a*x - b)*P[k,j] - c*P[k-1,j]

      else:
        P[0,j] = 0.0

        if n>0:
          if j==1:
            P[1,j] =  0.5*(alpha + beta + 2)
          else:
            P[1,j] = 0.0

        if n>1:
          for k in range(1, n):
            a, b, c = self.recursion_coefs(k, alpha, beta)

            P[k+1,j] = (a*x - b)*P[k,j] - c*P[k-1,j] +j*a*P[k,j-1]

    return P/nrm

  def eval_P(self, n, alpha, beta, x):
    P = np.zeros(shape=(n+1,) + np.shape(x))

    nrm = self.norm(np.arange(n+1), alpha, beta).reshape((-1,1))

    P[0] = 1.0

    if n>0:
      P[1] = 0.5*((alpha + beta + 2)*x \
             - (beta - alpha))

    if n>1:
      for k in range(1, n):
        a, b, c = self.recursion_coefs(k, alpha, beta)

        P[k+1] = (a*x - b)*P[k] - c*P[k-1]

    return P/nrm

  def vandermonde(self, n, m, alpha, beta, x):
    return self.eval_P_tensor(n, m, alpha, beta, x)[:,-1,:].T

  def forward_transform(self, u):
    """ Discrete Forward Jacobi transform u -> v where
        u(x) = sum_j v_j J^n_(alpha, beta)(x) and x are JGL nodes"""

    assert np.shape(u)[0]==self.n+1

    v = np.matmul(self.P[:,0] * self.weights, u)

    v[-1] = v[-1]/(2.0 + (self.alpha + self.beta + 1.0)/self.n)

    return v

  def backward_transform(self, v, x=None):
    """ Discrete Backward Jacobi transform v -> u where
        u(x) = sum_j v_j J^n_(alpha, beta)(x) """
    assert v.shape[0]==self.n+1

    if x is None:
      u = np.tensordot(self.P, v, axes=([0], [0]))
    else:
      P = self.evalJ(x)
      u = np.tensordot(P, v, axes=([0], [0]))

    return u
