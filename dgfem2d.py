import numpy as np
import itertools
from simplex2d import Simplex2D

class DGFEM2D(object):
  """ Discontinuous Galerkin FEM in two space dimensions"""
  def __init__(self, VX, VY, EtoV, N, NODETOL=1e-12):
    """
      VX: array of x coordinates of vertices, shape (Nv,1),
      VY: array of y coordinates of vertices, shape (Nv,1),
      EtoV: array of element indices, shape (K, Nfaces),
      N: degree of polynomial expansion in each element
    """
    self.NODETOL = NODETOL

    self.N = N

    self.Np = ((N+1)*(N+2))//2

    self.Nfp = N+1

    self.Nfaces = 3

    self.VX, self.VY = np.array(VX).reshape((-1,1)), np.array(VY).reshape((-1,1))
    assert len(VX)==len(VY)

    self.Nv = len(VX)

    self.K = len(EtoV)#np.shape(EtoV)[0]

    self.EtoV = EtoV - np.amin(EtoV)#np.amin(np.array(EtoV))

    self.basis = Simplex2D(N)

    r, s = self.basis.r, self.basis.s

    va = self.EtoV[:,0]
    vb = self.EtoV[:,1]
    vc = self.EtoV[:,2]

    self.X = 0.5*(-(r+s)*self.VX[va] + (1+r)*self.VX[vb] + (1+s)*self.VX[vc]).T
    self.Y = 0.5*(-(r+s)*self.VY[va] + (1+r)*self.VY[vb] + (1+s)*self.VY[vc]).T

    self.Fmask = np.zeros(shape=(self.Nfp, self.Nfaces), dtype=int)

    self.Fmask[:,0] = np.nonzero(np.fabs(s+1)<NODETOL)[0]
    self.Fmask[:,1] = np.nonzero(np.fabs(r+s)<NODETOL)[0]
    self.Fmask[:,2] = np.nonzero(np.fabs(r+1)<NODETOL)[0]

    self.Fx = self.X[self.Fmask.ravel(order='F'),:]
    self.Fy = self.Y[self.Fmask.ravel(order='F'),:]

    self.lift = self.lift()

    self.rx, self.xr, self.sx, self.xs,\
    self.ry, self.yr, self.sy, self.ys, self.J = self.geometric_factors()

    self.nx, self.ny, self.sJ = self.normals()

    self.Fscale = 1.0/self.J[self.Fmask.ravel(order='F'),:]

    self.EtoE, self.EtoF = self.connect()

    self.vmapM, self.vmapP, self.vmapB, self.mapB = self.build_maps()

  def geometric_factors(self):
    xr = np.matmul(self.basis.Dr, self.X)
    xs = np.matmul(self.basis.Ds, self.X)
    yr = np.matmul(self.basis.Dr, self.Y)
    ys = np.matmul(self.basis.Ds, self.Y)

    J = -xs*yr + xr*ys

    rx, sx = ys/J, -yr/J
    ry, sy = -xs/J, xr/J

    return rx, xr, sx, xs, ry, yr, sy, ys, J

  def normals(self):
    fxr = self.xr[self.Fmask.ravel(order='F'),:]
    fxs = self.xs[self.Fmask.ravel(order='F'),:]
    fyr = self.yr[self.Fmask.ravel(order='F'),:]
    fys = self.ys[self.Fmask.ravel(order='F'),:]

    fid1 = np.arange(self.Nfp)
    fid2 = np.arange(self.Nfp, 2*self.Nfp)
    fid3 = np.arange(2*self.Nfp, 3*self.Nfp)

    nx = np.zeros(shape=(self.Nfp*self.Nfaces, self.K))
    ny = np.zeros(shape=(self.Nfp*self.Nfaces, self.K))

    nx[fid1] = fyr[fid1]
    ny[fid1] = -fxr[fid1]

    nx[fid2] = fys[fid2] - fyr[fid2]
    ny[fid2] = -fxs[fid2] + fxr[fid2]

    nx[fid3] = -fys[fid3]
    ny[fid3] = fxs[fid3]

    sJ = np.sqrt(nx**2+ny**2)
    nx = nx/sJ
    ny = ny/sJ

    return nx, ny, sJ

  def lift(self):
    Emat = np.zeros(shape=(self.Np, self.Nfaces*self.Nfp))

    face1 = self.basis.r[self.Fmask[:,0]]
    #faceR = self.basis.r[self.Fmask[0:self.Nfp]]
    V1 = self.basis.vandermonde(self.N, 0, 0.0, 0.0, face1)
    massEdge1 = np.linalg.inv(np.matmul(V1, V1.T))
    Emat[self.Fmask[:,0], 0:self.Nfp] = massEdge1
    #Emat[self.Fmask[0:self.Nfp], 0:self.Nfp] = massEdge1

    face2 = self.basis.r[self.Fmask[:,1]]
    #faceR = self.basis.r[self.Fmask[self.Nfp:2*self.Nfp]]
    V2 = self.basis.vandermonde(self.N, 0, 0.0, 0.0, face2)
    massEdge2 = np.linalg.inv(np.matmul(V2, V2.T))
    Emat[self.Fmask[:,1], self.Nfp:2*self.Nfp] = massEdge2
    #Emat[self.Fmask[self.Nfp:2*self.Nfp], self.Nfp:2*self.Nfp] = massEdge2

    face3 = self.basis.s[self.Fmask[:,2]]
    #faceS = self.basis.s[self.Fmask[2*self.Nfp:]]
    V3 = self.basis.vandermonde(self.N, 0, 0.0, 0.0, face3)
    massEdge3 = np.linalg.inv(np.matmul(V3, V3.T))
    Emat[self.Fmask[:,2], 2*self.Nfp:3*self.Nfp] = massEdge3
    #Emat[self.Fmask[2*self.Nfp:], 2*self.Nfp:3*self.Nfp] = massEdge3

    lift = np.matmul(self.basis.V, np.matmul(self.basis.V.T, Emat))

    return lift

  def lift_op(self, fx, fy):
    pass    

  def grad(self, u):
    ur = np.matmul(self.basis.Dr, u)
    us = np.matmul(self.basis.Ds, u)

    ux = self.rx*ur + self.sx*us
    uy = self.ry*ur + self.sy*us

    return ux, uy

  def div(self, u, v):
    ur = np.matmul(self.basis.Dr, u)
    us = np.matmul(self.basis.Ds, u)
    vr = np.matmul(self.basis.Dr, v)
    vs = np.matmul(self.basis.Ds, v)

    divu = self.rx*ur + self.sx*us + self.ry*vr + self.sy*vs

    return divu

  def curl(self, ux, uy):
    uxr = np.matmul(self.basis.Dr, ux)
    uxs = np.matmul(self.basis.Ds, ux)
    uyr = np.matmul(self.basis.Dr, uy)
    uys = np.matmul(self.basis.Ds, uy)

    vz = self.rx*uyr + self.sx*uys - self.ry*uxr - self.sy*uxs

    return vz

  def lap(self, u):
    dxu, dyu = self.grad(u)
    return self.div(dxu, dyu)

  def connect(self):
    EtoE = np.zeros(shape=(self.K, self.Nfaces), dtype=int)
    EtoF = np.zeros(shape=(self.K, self.Nfaces), dtype=int)
    f2v = [[0,1],[1,2],[0,2]]

    for k1, f1 in itertools.product(range(self.K), range(self.Nfaces)):
      done = False
      for k2, f2 in itertools.product(range(self.K), range(self.Nfaces)):
        if len(np.intersect1d( self.EtoV[k1,f2v[f1]],
                               self.EtoV[k2,f2v[f2]] ))==2:
          if not done or k1!=k2:
            EtoE[k1,f1] = k2
            EtoF[k1,f1] = f2
            done = True

    return EtoE, EtoF

  def build_maps(self):
    vmapM = np.zeros(shape=(self.K, self.Nfaces, self.Nfp), dtype=int)
    kmapM = np.zeros(shape=(self.K, self.Nfaces, self.Nfp), dtype=int)
    vmapP = np.zeros(shape=(self.K, self.Nfaces, self.Nfp), dtype=int)
    kmapP = np.zeros(shape=(self.K, self.Nfaces, self.Nfp), dtype=int)

    for k in range(self.K):
      for f in range(self.Nfaces):
        vmapM[k, f, :] = self.Fmask[:, f]
        kmapM[k, f, :] = k

    vmapP[:] = vmapM
    kmapP[:] = kmapM

    for k1 in range(self.K):
      for f1 in range(self.Nfaces):
        k2 = self.EtoE[k1, f1]
        f2 = self.EtoF[k1, f1]

        for i1 in range(self.Nfp):
          x1 = self.X[vmapM[k1, f1, i1], k1]
          y1 = self.Y[vmapM[k1, f1, i1], k1]

          #x12 = self.X[vmapM[k1, (f1+1)%self.Nfaces, i1], k1]
          #y12 = self.Y[vmapM[k1, (f1+1)%self.Nfaces, i1], k1]
          x12 = self.X[vmapM[k1, f1, (i1+1)%self.Nfp], k1]
          y12 = self.Y[vmapM[k1, f1, (i1+1)%self.Nfp], k1]
          x13 = self.X[vmapM[k1, f1, (i1-1)%self.Nfp], k1]
          y13 = self.Y[vmapM[k1, f1, (i1-1)%self.Nfp], k1]

          refd = np.minimum(np.sqrt( (x1-x12)**2 + (y1-y12)**2 ),
                            np.sqrt( (x1-x13)**2 + (y1-y13)**2 ))

          for i2 in range(self.Nfp):
            x2 = self.X[vmapM[k2, f2, i2], k2]
            y2 = self.Y[vmapM[k2, f2, i2], k2]

            D = np.sqrt((x1-x2)**2 + (y1-y2)**2)

            if D<self.NODETOL:
              vmapP[k1, f1, i1] = vmapM[k2, f2, i2]
              kmapP[k1, f1, i1] = k2
            #if D<refd:
              #if k1!=k2:
                #vmapP[k1, f1, i1] = vmapM[k2, f2, i2]
                #kmapP[k1, f1, i1] = k2
                #break


    vmapM = vmapM.reshape((-1, self.K))
    vmapP = vmapP.reshape((-1, self.K))

    kmapM = kmapM.reshape((-1, self.K))
    kmapP = kmapP.reshape((-1, self.K))

    mapB = np.nonzero((vmapM==vmapP)*(kmapM==kmapP))
    vmapB = vmapM[mapB]
    kmapB = kmapM[mapB]
    vmapM = [vmapM, kmapM]
    vmapP = [vmapP, kmapP]
    vmapB = [vmapB, kmapB]

    return vmapM, vmapP, vmapB, mapB
