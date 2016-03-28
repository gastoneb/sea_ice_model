# world.py
# Graham Stonebridge
# Department of Systems Design Engineering
# University of Waterloo
# 2015
#
# 1D Model for Sea Ice Dynamics
# Fully implicit solution to ice momentum equations
# Spatial discretization on an Arakawa C-Grid
# Periodic in x-direction
# Forcing provided by a finite difference shallow water model on the same grid
#
#  C-Grid:
#  ----v----
#  |       |
#  u   a   u
#  |       |
#  ----v----
# All scalars defined at point 'a' including viscosity, thickness,
# concentration.
#

###############################################################################
# Import Libraries
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
import time
import scipy.ndimage.filters as filt
from utils import *

###############################################################################
# General Functions
###############################################################################

# delta_x (positive direction)
def dxp(f,dx):
    fx = (np.roll(f,-1) -f)/dx
    return fx

# delta_x (negative direction)
def dxm(f,dx):
    fx = (f - np.roll(f,1))/dx
    return fx

# average in x (positive direction)
def axp(f):
    afx = 0.5*(np.roll(f,-1) + f)
    return afx

# average in x (negative direction)
def axm(f):
    afx = 0.5*(f + np.roll(f,1))
    return afx

# Transport for a scalar located at the center of the C-grid (Forward Euler)
def transport(var, u, v, dt, dx, source):
    var_update = dt*(var/dt+(1/dx)*(u*(var+np.roll(var,-1))/2-np.roll(u,1)*
                 (var+np.roll(var,1))/2)+source)
    var_update = var + dxm(axp(var)*u,dx)*dt
    return var_update

###############################################################################
# Class describing the model domain
# Includes plotting functions
###############################################################################
class World(object):
    """A class containing general model parameters and functions """
    def __init__(self,fnam):
        # Option to read in parameters from an input file, "fnam".
        self.Lx = 1000e3
        self.Ly = 200e3
        self.Nx = 200
        self.Ny = 1
        self.dx = 2*self.Lx/self.Nx

        self.f0 = 0.0
        self.f0 = 1e-4
        self.t0 = 0
        self.tf = 3600*24*10
        self.tp = 3600
        self.t = np.copy(self.t0)
        self.fig = plt.figure(figsize=(8,8))
        self.grid = np.linspace(-self.Lx,self.Lx-self.dx,self.Nx)
        self.plot_bool = True # Set to true for plots on the fly.
        if self.plot_bool:
            plt.show(block=False)
            plt.ion()
        print(self.dx, self.Nx, self.grid.size)

    def figure_update(self,uw,u,a,h):
        if self.plot_bool:
            plt.clf()
            plt.subplot(4,1,1)
#            plt.plot(eta.T,'-b',linewidth=2, label='Ocean Surface')
#            plt.title('Ocean Surface Displacement (m)')
#            plt.ylim(-5,15)
            plt.plot(uw.T,'-b',linewidth=2, label='Ocean Surface')
            plt.title('Ocean velocity (m/s)')
            plt.ylim(-0.3,0.3)
            plt.subplot(4,1,2)
            plt.plot(u.T,'-r',linewidth=2, label='Ice u-velocity')
            plt.title('Ice Velocity (m/s)')
            plt.ylim([-0.2,0.2])
            plt.subplot(4,1,3)
            plt.plot(h.T,'-or', linewidth=2, label='Ice thickness')
            plt.title('Thickness (m)')
            plt.subplot(4,1,4)
            plt.plot(a.T,'-or', linewidth=2, label='Ice concentration')
            plt.title('Concentration (0-1)')
            plt.show(block=False)
            plt.pause(0.0001)
##            plt.tight_layout()
            plt.draw()

    def figure_final(self,eta,u,a,h):
        if self.plot_bool:
            plt.ioff()
            plt.subplot(4,1,1)
            plt.plot(eta.T,'-ob',linewidth=2, label='Ocean Surface')
            plt.title('Ocean Surface Displacement (m)')
            plt.subplot(4,1,2)
            plt.plot(u.T,'-ob',linewidth=2, label='Ice u-velocity')
            plt.title('U-Velocity (m/s)')
            plt.subplot(4,1,3)
            plt.plot(h.T,'-or', linewidth=2, label='Ice thickness')
            plt.title('Thickness (m)')
            plt.subplot(4,1,4)
            plt.plot(a.T,'-or', linewidth=2, label='Ice concentration')
            plt.title('Concentration (0-1)')
            plt.show()


###############################################################################
# Class for ocean
###############################################################################
class Ocean(object):
    """A Class containing ocean state variables and numerical methods"""
    def __init__(self,parms):
        print("Instantiating Ocean Class")
        self.Nx = np.copy(parms.Nx)
        self.Ny = 1
        self.dx = parms.Lx/self.Nx
        self.dy = parms.Ly/self.Ny
        self.x = np.zeros((self.Ny,self.Nx))+np.linspace(-parms.Lx/2,parms.Lx/2-self.dx,self.Nx)
        self.u = np.ones((self.Ny,self.Nx))*(0.0)
        self.v = np.zeros((self.Ny,self.Nx))
        self.H0 = 100
        self.dt = 120
        self.rhow = 1035
        self.gp = 9.81/100
        self.f0 = 0.0#parms.f0
#        self.method = "sw_ener" # or sw, const
        self.method=self.flux_sw_ener
#        np.random.seed(10)
        self.eta=15*np.random.randn(self.Ny,self.Nx)
        self.eta=filt.gaussian_filter(self.eta,10,mode='wrap')
#        self.eta += 5*np.exp(-(self.x**2)/(parms.Lx/8)**2) #Gaussian
        self.flux_prev1 = np.zeros((3,self.Nx))
        self.flux_prev2 = np.zeros((3,self.Nx))
        self.Lx = np.copy(parms.Lx)

    def flux_const(self):
        f1=self.u*0.
        f2=self.v*0.
        f3=self.eta*0.
        flux=np.vstack([f1,f2,f3])
        return flux

    def restart(self):
        self.u = self.u/self.u*0.1
        self.v = self.v*0.
        self.eta = 15*np.random.randn(self.Ny,self.Nx)
        self.eta=filt.gaussian_filter(self.eta,4,mode='wrap')
#        self.eta += 5*np.exp(-(self.x**2)/(self.Lx/8)**2) #Gaussian
        self.flux_prev1 = np.zeros((3,self.Nx))
        self.flux_prev2 = np.zeros((3,self.Nx))

    def perturb_state(self):
        v = 5*np.random.randn(self.Ny,self.Nx)
        w = filt.gaussian_filter(v,4,mode='wrap')
        self.eta += w

    # Compute energy-conserving flux for shallow water equations
    def flux_sw_ener(self):
        # author: Francis Poulin, University of Waterloo, 2015

        # Compute Fields
        h = self.H0  +  self.eta
        U = axp(h)*self.u
        V = h*self.v
        B = self.gp*h + 0.5*(axm(self.u**2) + self.v**2)
        q = (dxp(self.v,self.dx) + self.f0)/axp(h)

        # Compute fluxes
        flux = np.vstack([ q*axp(V) - dxp(B,self.dx), -axm(q*U), -dxm(U,self.dx)])
        return flux

    # Forward Euler time stepping
    def time_step(self):
        flux = self.method()
        self.u = self.u + self.dt/12*(23*flux[0,:]-16*self.flux_prev1[0,:]+5*self.flux_prev2[0,:])
        self.v = self.v + self.dt/12*(23*flux[1,:]-16*self.flux_prev1[1,:]+5*self.flux_prev2[1,:])
        self.eta = self.eta + self.dt/12*(23*flux[2,:]-16*self.flux_prev1[2,:]+5*self.flux_prev2[2,:])
        self.flux_prev1, self.flux_prev2 = flux, self.flux_prev1

###############################################################################
# Class for the atmosphere (EMPTY FOR NOW)
###############################################################################
class Atmosphere(object):
    """A class containing atmospheric state variables and numerical methods"""
    def __init__(self,parms):
        print("Instantiating Atmosphere Class")
        self.Nx = np.copy(parms.Nx)
        self.Ny = 1
        self.dx = parms.Lx/self.Nx
        self.dy = parms.Ly/self.Ny
        self.u = np.zeros((self.Nx,self.Ny))
        self.v = np.zeros((self.Nx,self.Ny))
        self.dt = 3600*2
        self.rhoa = 1.3

###############################################################################
# Class for sea ice
###############################################################################
class Ice(object):
    """A class containing sea ice state variables and numerical methods"""
    def __init__(self,parms):
        print("Instantiating Ice Class")
        self.grid_model = parms.grid.reshape((parms.Nx,1))
        self.Nx = np.copy(parms.Nx)
        self.Ny = 1
        self.Lx = parms.Lx
        self.dx = parms.Lx/self.Nx
        self.dy = parms.Ly/self.Ny
        self.u = np.zeros((self.Ny,self.Nx))
        self.v = np.zeros((self.Ny,self.Nx))
        self.a = np.ones((self.Ny,self.Nx))*0.99
        self.h = np.ones((self.Ny,self.Nx))*0.4  #Initial thickness
        self.dt = 3600*2
        self.n_outer_loops = 10
        self.e = 2
        self.Cw = 0.0055
        self.Ca = 0.0012
        self.Ps = 5000
        self.C = 20
        self.theta = 25*np.pi/180
        self.phi = 25*np.pi/180
        self.rhoi = 991
        self.rhow = 1035
        self.eta = np.zeros((self.Ny,self.Nx))
        self.zeta = np.zeros((self.Ny,self.Nx))
        self.f0 = parms.f0
        self.flux_prev1 = np.zeros((2,self.Ny,self.Nx))
        self.flux_prev2 = np.zeros((2,self.Ny,self.Nx))
        self.n_outer_loops = 10
        self.s_h = np.zeros((self.Ny,self.Nx))
        self.s_a = np.zeros((self.Ny,self.Nx))
        self.uprev = np.copy(self.u)

    # Construct b column vector of Ax=b
    def build_b(self,uw,vw):
        bu = -((self.rhoi/2/self.dt)*(np.roll(self.a,-1)*np.roll(self.h,-1)+self.a*self.h)*self.u+
            (self.rhow*self.Cw/2)*(self.a+np.roll(self.a,-1))*np.absolute(uw-self.u)*(uw))
        return bu

    # Construct A matrix of Ax=b
    def build_A(self,uw,vw):
        # A is a sparse matrix (Nx*Ny*2)x(Nx*Ny*2)
#        A=sparse.dok_matrix((self.Nx*self.Ny*2,self.Nx*self.Ny*2))

        # Compute coeffcients of A
        cu1 = 1/(self.dx**2)*(self.zeta)
        cu2 = (-1/(self.dx)**2*(np.roll(self.zeta,-1)+self.zeta)-
                self.rhoi/2/self.dt*(np.roll(self.a*self.h,-1)+self.a*self.h)-
                self.rhow*self.Cw/2*(self.a+np.roll(self.a,-1))*np.absolute(uw-self.u))
        cu3 = 1/(self.dx**2)*(np.roll(self.zeta,-1))

        A11 = sparse.spdiags(np.vstack((np.roll(np.ravel(cu1).T,-1),np.ravel(cu2).T,np.roll(np.ravel(cu3).T,1))),[-1,0,1],self.Nx*self.Ny,self.Nx*self.Ny).todok()
        A11[self.Nx*self.Ny-1,0]=cu1[0,0]
        A11[0,self.Nx*self.Ny-1]=cu3[0,self.Nx*self.Ny-1]
        A = A11
        return A

    # Solve for x in Ax=b after setting up equations. x is the model state.
    def solve_momentum_equations(self,uw,vw):
        self.update_viscosities()
        b = self.build_b(uw,vw).T
        A = self.build_A(uw,vw)
        xprev = np.ravel(self.u)
        # Solve using a sparse solver in scipy.sparse.linalg (any solver works)
#        [x,result] = sparse.linalg.bicgstab(A,b,x0=xprev,tol=0.01)
#        [x,result] = sparse.linalg.bicgstab(A,b, x0=xprev)
        x = sparse.linalg.spsolve(A,b)
        # solving the system directly is fastest because the system is quite small
        self.u = np.reshape(x[0:self.Nx*self.Ny],(self.Ny,self.Nx))

    # Compute eta and zeta
    def update_viscosities(self):
        P = self.Ps*self.h*np.exp(-self.C*(1-self.a))
        Delta = np.sqrt(((self.u-np.roll(self.u,1))/self.dx)**2*(1+self.e**(-2)))+1e-32
        self.zeta = P/(2*Delta)
        # upper and lower bound on viscosity per Hibler 1979
        maxzeta=(P/4)*10**9
        minzeta=4*10**8
        self.zeta=np.minimum(self.zeta,maxzeta)
        self.zeta=np.maximum(self.zeta,minzeta)
#        self.eta = self.zeta/self.e**2
        self.eta = self.zeta*0.0

      # Correct a and h for non-physical values
    def redistribution(self):
        #Correct a > 1
        tmp = np.copy(self.a)
        tmp[tmp>0.999]=0.999
        errA=self.a-tmp
        self.h += self.h*errA
        self.a = self.a - errA

        #Correct a < 0
        tmp = np.copy(self.a)
        tmp[tmp<0.1]=0.1
        self.a=np.copy(tmp)

        #Correct h<0
        tmp=np.copy(self.h)
        tmp[tmp<0.1]=0.1
        self.h=np.copy(tmp)

    # Hibler 1979's growth terms
    def update_thermodynamics(self):

        h0 = 0.1
        g0 = self.growth(self.h*0.0)
        # thickness growth term
        self.s_h = self.growth(self.h/self.a)*self.a+(1-self.a)*g0

        # concentration growth term
        g1 = g0/h0*(1-self.a)
        g4 = (self.a/2/self.h)*self.s_h

        self.s_a = self.h*0.0
        self.s_a[g0>0] = g1[g0>0]
        self.s_a[self.s_h<0] = g4[self.s_h<0]

    # Compute growth rate as a function of ice thickness and season
    def growth(self,h):
        month = 1
        if month == 1:
            r,s,a = 1.0, 0.0, 0.12
        elif month == 7:
            r,s,a = 3,-0.01,-0.015

        # 'a' represents the maximum daily growth rate in m/day.
        a  = a*0.5*np.ones(h.shape)

        g = (a+(s-a)*(1.-1*np.exp(-3*h**1/r**1)))/24/60/60
        g *= 0.5
        return g

    # March solution forward one time step
    def time_step(self,ocean_u,ocean_v):
        # Perturb ocean forcing
        uw, vw = np.copy(ocean_u), np.copy(ocean_v)

        # Outer Loop
        for i in range(0,self.n_outer_loops):
            # Inner Iterations
            self.solve_momentum_equations(uw,vw)
        self.uprev = np.copy(self.u)

        # Compute thermodynamic growth/melt source terms
        self.update_thermodynamics()

        # Integrate advection eqn for h
        flux_h = -dxm(axp(self.h)*self.u,self.dx) + self.s_h
        self.h = self.h + self.dt/12*(23*flux_h-16*self.flux_prev1[0,:,:]+5*self.flux_prev2[0,:,:])
        self.flux_prev2[0,:,:], self.flux_prev1[0,:,:] = self.flux_prev1[0,:,:], flux_h

        # Integrate advection eqn for a
        flux_a = -dxm(axp(self.a)*self.u,self.dx) + self.s_a
        self.a = self.a + self.dt/12*(23*flux_a-16*self.flux_prev1[1,:,:]+5*self.flux_prev2[1,:,:])
        self.flux_prev2[1,:,:], self.flux_prev1[1,:,:] = self.flux_prev1[1,:,:], flux_a

        # Compute thickness
        self.logh = np.log10(self.h)

        # Correct for any non-physical concentration or thicknesses
        self.redistribution()
