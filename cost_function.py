import numpy as np
from numpy import linalg as LA
from scipy.linalg import sqrtm as SQRTM

def func_L2(z,f,g,mu):
    z = np.reshape(z,(len(z),1))
    f = LA.norm(f-np.dot(g,z))**2+(mu**2)*(LA.norm(z)**2)
    #print(f)  
    return f
    
def func_L1(z,f,g,mu,D,C_B,x_b,delta):
    z = np.reshape(z,(len(z),1)) 
    f = LA.norm(f-np.dot(g,z))**2+(mu**2)*(LA.norm(z)**2)
    f += LA.norm(np.dot(D,np.dot(LA.cholesky(C_B),z)+x_b),1)
    #z=np.matrix(z.reshape(n,1))   
#    print f
    return f
    
def grad_func_L2(z,f,g,mu):
    z = np.reshape(z,(len(z),1)) 
    f = -2*np.dot(g.T,(f-np.dot(g,z))) + 2*(mu**2)*z
    return f.flatten()

def grad_func_L1(z,f,g,mu,D,C_B,x_b,delta):
    z = np.reshape(z,(len(z),1)) 
    f = -2*np.dot(g.T,(f-np.dot(g,z))) + 2*(mu**2)*z + delta*np.dot(np.dot(LA.cholesky(C_B),D.T),np.sign(z))
    return f.flatten()

def Hess_func_L2(z,f,g,mu):
    return 2*np.dot(g.T,g)+2*mu**2
    
def Hess_func_L1(z,f,g,mu,D,C_B,x_b,delta):
    return 2*np.dot(g.T,g)+2*mu**2
