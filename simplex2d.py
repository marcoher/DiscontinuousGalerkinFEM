from __future__ import division
import numpy as np
from jacobi import Jacobi

class Simplex2D(Jacobi):
  """ Orthonormal basis on an equilateral triangle and related methods """
  def __init__(self, N):
    self.N = N

    self.Np = ((N + 1) * (N + 2)) // 2

    self.x, self.y = self.nodes_2D(N)

    self.r, self.s = self.xy2rs(self.x, self.y)

    self.V, self.Vr, self.Vs = self.vandermonde2D(N, self.r, self.s)

    self.Dr, self.Ds = self.dmats2D(self.V, self.Vr, self.Vs)

    self.invV = np.linalg.inv(self.V)

    self.Mass = np.matmul(self.invV.T, self.invV)

  def simplex2DP(self, a, b, i, j):
    """ evaluates basis_(i,j) at a, b, which may be vectors"""
    h1 = self.eval_P(i, 0.0, 0.0, a)[-1]
    h2 = self.eval_P(j, 2*i+1.0, 0.0, b)[-1]

    P = np.sqrt(2.0) * h1 * h2 * (1.0 - b)**i

    return  P

  def grad_simplex2DP(self, a, b, id, jd):
    Pa = self.eval_P_tensor(id, 1, 0.0 ,0.0, a)
    Pb = self.eval_P_tensor(jd, 1, 2*id+1, 0, b)

    fa, dfa = Pa[-1,0,:], Pa[-1,1,:]
    gb, dgb = Pb[-1,0,:], Pb[-1,1,:]

    #fa = self.eval_P(id, 0.0 ,0.0, a)
    #dfa = self.grad_eval(id, 0.0, 0.0, a)
    #gb = self.eval_P(jd, 2*id+1.0, 0.0, b)
    #dgb = self.grad_eval(jd, 2*id+1.0, 0.0, b)

    dmodedr = dfa*gb
    if id>0:
      dmodedr = dmodedr*(0.5*(1-b))**(id-1)

    dmodeds = dfa*(gb*(0.5*(1+a)))
    if id>0:
      dmodeds = dmodeds*(0.5*(1-b))**(id-1)

    tmp = dgb*(0.5*(1-b))**id
    if id>0:
      tmp = tmp - 0.5*id*gb*(0.5*(1-b))**(id-1)
    dmodeds = dmodeds + fa*tmp

    dmodedr = 2**(id+0.5)*dmodedr
    dmodeds = 2**(id+0.5)*dmodeds

    return dmodedr, dmodeds

  def rs2ab(self, r, s):

    #a = np.where(s<1.0, 2.0*(1.0 + r)/(1.0 - s) - 1.0, -1.0)
    a = 2*(1.0 + r)/((s<1.0) - s) -1
    b = s

    return a, b

  def xy2rs(self, x, y):
    L1 = (np.sqrt(3)*y + 1.0)/3
    L2 = (-3*x - np.sqrt(3)*y + 2.0)/6
    L3 = (3*x - np.sqrt(3)*y + 2.0)/6

    r = -L2 + L3 - L1
    s = -L2 - L3 + L1

    return r, s

  def warp_factor(self, n, r_out):
    LGL_r, _ = self.gauss_lobatto_nodes(n, 0.0 ,0.0)

    r_eq = np.linspace(-1.0, 1.0, n+1)
    #r_eq = np.reshape(np.linspace(-1.0, 1.0, N+1), (-1,))

    V_eq = self.vandermonde(n, 0, 0.0, 0.0, r_eq)

    nr = len(r_out)

    P_mat = self.eval_P_tensor(n, 0, 0, 0, r_out)[:,0,:]

    #P_mat = np.zeros(shape=(n+1, nr))
    #for i in range(n+1):
    #  Pmat[i,:] = self.eval_P(i, 0, 0, 0, rout)

    L_mat = np.linalg.solve(V_eq.T, P_mat)

    warp = np.matmul(L_mat.T, LGL_r - r_eq)

    zerof = (np.fabs(r_out)<(1.0-1.0e-10))

    sf = 1.0 -(zerof*r_out)**2

    warp = warp/sf + warp*(zerof-1)

    return warp

  def nodes_2D(self, N):
    alpopt = [0.0000, 0.0000, 1.4152, 0.1001, 0.2751, 0.9800, 1.0999,\
              1.2832, 1.3648, 1.4773, 1.4959, 1.5743, 1.5770, 1.6223, 1.6258]
    if N<16:
      alpha = alpopt[N]
    else:
      alpha = 5.0/3.0

    Np = ((N + 1)*(N + 2))//2

    L1 = np.zeros(shape=(Np,))
    L2 = np.zeros(shape=(Np,))
    L3 = np.zeros(shape=(Np,))

    sk = 0
    for n in range(N+1):
      for m in range(N+1-n):
        L1[sk] = (n+0.0)/N
        L3[sk] = (m+0.0)/N
        sk = sk + 1

    L2 = 1.0 - L1 - L3
    x = -L2 + L3
    y = (-L2 - L3 + 2*L1)/np.sqrt(3.0)

    blend1 = 4*L2*L3
    blend2 = 4*L1*L3
    blend3 = 4*L1*L2

    warpf1 = self.warp_factor(N, L3 - L2)
    warpf2 = self.warp_factor(N, L1 - L3)
    warpf3 = self.warp_factor(N, L2 - L1)

    warp1 = blend1 * warpf1 * (1.0 + (alpha*L1)**2)
    warp2 = blend2 * warpf2 * (1.0 + (alpha*L2)**2)
    warp3 = blend3 * warpf3 * (1.0 + (alpha*L3)**2)

    x = x + warp1 + np.cos(2*np.pi/3)*warp2 + np.cos(4*np.pi/3)*warp3
    y = y + np.sin(2*np.pi/3)*warp2 + np.sin(4*np.pi/3)*warp3

    return x, y

  def vandermonde2D(self, N, r, s):
    Np = ((N+1)*(N+2))//2
    V2D = np.zeros(shape=(len(r), Np))
    V2Dr = np.zeros(shape=(len(r), Np))
    V2Ds = np.zeros(shape=(len(s), Np))

    a, b = self.rs2ab(r, s)
    sk = 0
    for i in range(N+1):
      for j in range(N-i+1):
        V2D[:,sk] = self.simplex2DP(a, b, i, j)
        V2Dr[:,sk], V2Ds[:,sk] = self.grad_simplex2DP(a, b, i, j)
        sk = sk + 1

    return V2D, V2Dr, V2Ds

  def dmats2D(self, V, Vr, Vs):
    VT = np.transpose(V)
    Dr = np.transpose(np.linalg.solve(VT, np.transpose(Vr)))
    Ds = np.transpose(np.linalg.solve(VT, np.transpose(Vs)))

    return Dr, Ds

  def filter2D(self, Norder, Nc, sp):
    Norderp = (Norder+1)*(Norder+2)//2
    filterdiag = np.ones(shape =(Norderp,1))
    alpha = -np.log(self.eps)

    sk = 0
    for i in range(Norder+1):
      for j in range(Norder-i+1):
        if i+j>=Nc:
          filterdiag[sk] = np.exp(-alpha*((i+j-Nc)/(Norder-Nc))**sp)
        sk = sk + 1

    tmp = np.transpose( np.matmul(V, np.diag(filterdiag)) )
    F = np.transpose( np.linalg.solve(np.transpose(V), tmp) )

    return F
