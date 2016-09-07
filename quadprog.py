# -*- coding: utf-8 -*-
"""
Created on Wed Jul 06 16:36:48 2016

@author: Nazanin
"""
import numpy as np
from numpy import linalg as LA
from scipy.optimize import minimize
from cvxopt import matrix
from cvxopt import solvers

def solve(z,f,G,mu,D,C_B,x_b,delta):
    N = z.size
        
    v = delta*np.dot(D,np.dot(LA.cholesky(C_B),z)+x_b)
    v_pos = np.maximum(v,0)
    v_neg = np.maximum(-v,0)
    
    w0 = np.concatenate((z,v_pos,v_neg),axis=0)
    
    H = np.zeros([3*N,3*N])
    H[:N,:N] = 2*np.dot(G.T,G)+mu**2*np.ones([N,N])
    
    c = np.ones([3*N,1])
    c[:N] = -2*np.dot(G.T,f)
        
    E = np.zeros([N,3*N])
    E[:,:N] = delta*np.dot(D,LA.cholesky(C_B))
    E[:,N:2*N] = -np.eye(N)
    E[:,2*N:] = np.eye(N)
    
    F = np.zeros(H.shape)
    F[N:2*N,N:2*N] = F[2*N:,2*N:] = np.eye(N)
    
    g = -delta*np.dot(D,x_b) 
    
    func_L1 = lambda w: np.asscalar(0.5*np.dot(np.dot(np.vstack(w).T,H),np.vstack(w))+np.dot(c.T,np.vstack(w)))
    
    cons = [{'type': 'ineq','fun':lambda w: np.dot(F,np.vstack(w)).flatten()},{'type': 'eq','fun':lambda w: (np.dot(E,np.vstack(w))-g).flatten()}]

    opt = {'disp':True,'ftol': 1e-01,'maxiter': 5}

    res_cons = minimize(func_L1, w0, constraints=cons, method='SLSQP', options=opt)
    
    sol = res_cons.x
    
#    h = np.zeros([3*N,1])    
#        
#    res = solvers.qp(matrix(H),matrix(c),matrix(-F),matrix(h),matrix(E),matrix(g),solver='MOSEK')
#    
#    sol = res['x']
                                 
    return sol[:1000]