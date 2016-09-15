# world.py
# Graham Stonebridge
# Department of Systems Design Engineering
# University of Waterloo
# 2016
#
# 1D Model for Sea Ice Dynamics

###############################################################################
# Import Libraries
###############################################################################

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
import time
import scipy.ndimage.filters as filt
from utils import *

###############################################################################
# Main class with some global parameters.
###############################################################################
class Model():
    """A class containing general model parameters and functions """
    def __init__(self):
        self.Lx = 500e3
        self.Ly = 200e3
        self.Nx = 1000
        self.Ny = 1
        self.dx = 2*self.Lx/self.Nx
        self.dy = self.Ly/self.Ny
        self.f0 = 0.0
        self.t0 = 0
        self.tf = 3600*24*30 
        self.tp = 3600
        self.t = np.copy(self.t0)
        self.grid = np.linspace(-self.Lx,self.Lx-self.dx,self.Nx)
        self.dist = distance_periodic(self.grid.reshape((self.Nx,1)),self.Lx)
        self.plot_bool = True # Set to true for plots on the fly.
        self.rhow = 1035
        self.rhoi = 991
        self.rhoa = 1.3
        self.gp = 9.81/100

###############################################################################
# Class for ocean
###############################################################################
class Ocean(Model):
    """A Class containing ocean state variables and numerical methods"""
    def __init__(self):
        print("Instantiating Ocean Class")
        Model.__init__(self)
        self.u = np.ones((self.Ny,self.Nx))*(0.0)
        self.v = np.zeros((self.Ny,self.Nx))
        self.H0 = 100
        self.dt = 1
        self.method = self.flux_sw_ener
        self.length_scale = 100000
        self.eta_variance = 10
        self.distribution = 'gaussian'
        self.eta = gen_SRF(gen_covmatrix(self.dist,self.length_scale,self.eta_variance,self.distribution))
        self.eta_error_variance = 2
        self.flux_prev1 = np.zeros((3,self.Nx))
        self.flux_prev2 = np.zeros((3,self.Nx))
        self.time_scaling = 0.25 #Slow down / speed up the dynamics

    def flux_const(self): # For a steady state simulation
        f1=self.u*0.
        f2=self.v*0.
        f3=self.eta*0.
        flux=np.vstack([f1,f2,f3])
        return flux

    def restart(self):
        self.u = self.u*0.
        self.v = self.v*0.
        self.eta = gen_SRF(gen_covmatrix(self.dist,self.length_scale,self.eta_variance,'gaussian'))
        self.flux_prev1 = np.zeros((3,self.Nx))
        self.flux_prev2 = np.zeros((3,self.Nx))

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
        self.u = self.u + self.dt*self.time_scaling/12*(23*flux[0,:]-16*self.flux_prev1[0,:]+5*self.flux_prev2[0,:])
        self.v = self.v + self.dt*self.time_scaling/12*(23*flux[1,:]-16*self.flux_prev1[1,:]+5*self.flux_prev2[1,:])
        self.eta = self.eta + self.dt*self.time_scaling/12*(23*flux[2,:]-16*self.flux_prev1[2,:]+5*self.flux_prev2[2,:])
        self.flux_prev1, self.flux_prev2 = flux, self.flux_prev1

###############################################################################
# Class for the atmosphere (same SW model as the ocean class)
###############################################################################
class Atmosphere(Model):
    """A class containing atmospheric state variables and numerical methods"""
    def __init__(self):
        print("Instantiating Atmosphere Class")
        Model.__init__(self)
        self.u = np.zeros((self.Ny,self.Nx))
        self.v = np.zeros((self.Ny,self.Nx))
        self.H0 = 100
        self.dt = 1
        self.method = self.flux_sw_ener
        self.length_scale = 50000
        self.eta_variance = 1 
        self.eta_error_variance = 2
        self.eta = gen_SRF(gen_covmatrix(self.dist,self.length_scale,self.eta_variance,'gaussian'))
        self.flux_prev1 = np.zeros((3,self.Nx))
        self.flux_prev2 = np.zeros((3,self.Nx))
        self.time_scaling = 0.25 #Slow down / speed up the dynamics

    def flux_const(self): # For a steady state simulation
        f1=self.u*0.
        f2=self.v*0.
        f3=self.eta*0.
        flux=np.vstack([f1,f2,f3])
        return flux

    def restart(self):
        self.u = self.u*0.
        self.v = self.v*0.
        self.eta = gen_SRF(gen_covmatrix(self.dist,self.length_scale,self.eta_variance,'gaussian'))
        self.flux_prev1 = np.zeros((3,self.Nx))
        self.flux_prev2 = np.zeros((3,self.Nx))

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
        self.u = self.u + self.dt*self.time_scaling/12*(23*flux[0,:]-16*self.flux_prev1[0,:]+5*self.flux_prev2[0,:])
        self.v = self.v + self.dt*self.time_scaling/12*(23*flux[1,:]-16*self.flux_prev1[1,:]+5*self.flux_prev2[1,:])
        self.eta = self.eta + self.dt*self.time_scaling/12*(23*flux[2,:]-16*self.flux_prev1[2,:]+5*self.flux_prev2[2,:])
        self.flux_prev1, self.flux_prev2 = flux, self.flux_prev1

###############################################################################
# Class for sea ice
###############################################################################
class Ice(Model):
    """A class containing sea ice state variables and numerical methods"""
    def __init__(self):
        Model.__init__(self)
        print("Instantiating Ice Class")
        self.dt = 3600*0.5
        # Initial conditions
        self.u = np.zeros((self.Ny,self.Nx))
        self.v = np.zeros((self.Ny,self.Nx))
        self.a = np.ones((self.Ny,self.Nx))*0.9
        self.h = np.ones((self.Ny,self.Nx))*0.1
        # Parameters
        self.e = 2
        self.Cw = 0.0055
        self.Ca = 0.0012
        self.Ps = 1500
        self.C = 20
        self.eta = np.zeros((self.Ny,self.Nx))
        self.zeta = np.zeros((self.Ny,self.Nx))
        self.n_outer_loops = 10
        # Other variables
        self.flux_prev1 = np.zeros((2,self.Ny,self.Nx))
        self.flux_prev2 = np.zeros((2,self.Ny,self.Nx))
        self.s_h = np.zeros((self.Ny,self.Nx))
        self.s_a = np.zeros((self.Ny,self.Nx))
        self.uprev = np.copy(self.u)
        self.growth_scaling = np.ones(self.h.shape)

    # Construct b column vector of Ax=b
    def build_b(self,uw,ua,uprev):
        b = -((self.rhoi/2/self.dt)*(np.roll(self.a,-1)*np.roll(self.h,-1)+self.a*self.h)*uprev+
            (self.rhow*self.Cw/2)*(self.a+np.roll(self.a,-1))*np.absolute(uw-uprev)*(uw)+
             (self.rhoa*self.Ca/2)*(self.a+np.roll(self.a,-1))*np.absolute(ua)*ua)
        return b

    # Construct A matrix of Ax=b
    def build_A(self,uw,ua,uprev):
        visc = self.zeta*(1+self.e**(-2))
        # Compute coeffcients of A
        cu1 = 1/(self.dx**2)*(visc)
        cu2 = (-1/(self.dx)**2*(np.roll(visc,-1)+visc)-
                self.rhoi/2/self.dt*(np.roll(self.a*self.h,-1)+self.a*self.h)-
                self.rhow*self.Cw/2*(self.a+np.roll(self.a,-1))*np.absolute(uw-uprev))
        cu3 = 1/(self.dx**2)*(np.roll(visc,-1))

        A = sparse.spdiags(np.vstack((np.roll(np.ravel(cu1).T,-1),np.ravel(cu2).T,np.roll(np.ravel(cu3).T,1))),[-1,0,1],self.Nx*self.Ny,self.Nx*self.Ny).todok()
        A[self.Nx*self.Ny-1,0]=cu1[0,0]
        A[0,self.Nx*self.Ny-1]=cu3[0,self.Nx*self.Ny-1]
        return A.tocsc()

    # Solve for x in Ax=b after setting up equations
    def solve_momentum_equations(self,uw,ua,uprev):
        self.update_viscosities()
        b = self.build_b(uw,ua,uprev).T
        A = self.build_A(uw,ua,uprev)
        xprev = np.ravel(uprev)
        # Can use any sparse solver in np.linalg but directly solving the system of
        # equations turns out to be the fastest.
#        x = sparse.linalg.spsolve(A,b)
        [x,junk] = sparse.linalg.cg(A,b,xprev,maxiter=500)
        self.u = np.reshape(x[0:self.Nx*self.Ny],(self.Ny,self.Nx))

    # Compute eta and zeta
    def update_viscosities(self):
        P = self.Ps*self.h*np.exp(-self.C*(1-self.a))
        Delta = np.sqrt(((self.u-np.roll(self.u,1))/self.dx)**2*(1+self.e**(-2)))+1e-32
        self.zeta = P/(2*Delta)
        # upper and lower bounds on viscosity
        maxzeta=(P/4)*10**9
        minzeta=4*10**8
        self.zeta[self.zeta<minzeta] = minzeta
        self.zeta[self.zeta>maxzeta] = maxzeta[self.zeta>maxzeta]

      # Correct a and h for non-physical values
    def redistribution(self):
        #Correct a > 1
        tmp = np.copy(self.a)
        tmp[tmp>1]=1
        errA=self.a-tmp
        self.h += self.h*errA
        self.a = self.a - errA

        #Correct a < 0
        tmp = np.copy(self.a)
        tmp[tmp<0.05]=0.05
        self.a=np.copy(tmp)

        #Correct h<0
        tmp=np.copy(self.h)
        tmp[tmp<0.1]=0.1
        self.h=np.copy(tmp)

    # Hibler 1979's growth terms
    def update_thermodynamics(self):
        h0 = 0.05
        g0 = self.growth(self.h*0.0)
        self.s_h = self.growth(self.h/self.a)*self.a+(1-self.a)*g0
        g1 = g0/h0*(1-self.a)
        g4 = (self.a/2/self.h)*self.s_h
        self.s_a = self.h*0.0
        self.s_a[g0>0] = g1[g0>0]
        self.s_a[self.s_h<0] = g4[self.s_h<0]

    # Compute growth rate as a function of ice thickness and season
    def growth(self,h):
        month = 1
        if month == 1:
            r,s,a = 1.5, 0.0, 0.12
        elif month == 7:
            r,s,a = 3,-0.01,-0.015
        a  = a*np.ones(h.shape)
        g = (a+(s-a)*(1.-1*np.exp(-3*h**1/r**1)))/24/60/60
        return g*self.growth_scaling

    # March solution forward one time step
    def time_step(self,uw,u_atm):
        ua = u_atm*150 #Multiply the wind velocity by some factor that makes it realistic.
        # Outer Loop
        uprev = np.copy(self.u)
        for i in range(0,self.n_outer_loops):
            # Inner Iterations
            self.solve_momentum_equations(uw,ua,uprev)
            uprev = (self.u+uprev)/2
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

        # Correct for any non-physical concentration or thicknesses
        self.redistribution()
