import numpy as np
#from geostatistics import *

# Sigmoid transformation for sea ice concentration
# FWD converts concentration [0-1] to [-inf -> inf]
# BKWD converts the transformed concentration back to [0-1]
def sigmoid_fwd(a):
    a[a>=1] = 0.999
    a[a<= 0] = 0.001
    return -np.log(1/a-1)

def sigmoid_bkwd(s):
    a = 1/(1+np.exp(-s))
    return a

# return minimum distance between points on a periodic 1D grid
def distance_periodic(x,Lx):
    nx = np.size(x)
    X = np.array([x[0,:],]*nx)
    X = np.tile(x,nx)
    distance_1 = np.abs(X-X.T)
    distance_2 = np.abs(X+2*Lx-X.T)
    distance_3 = np.abs(X-2*Lx-X.T)
    D = np.minimum(distance_1, distance_2)
    D = np.minimum(D, distance_3)
    return D


def gen_covmatrix(d,r,s,type):
    # d must be a square matrix of distances between points
    if d.shape[0] != d.shape[1]:
        print('Distance matrix is the wrong size',d.shape)
    # compute semivariance, then compute covariance matrix Q
    if type == 'gaussian':
        gamma = gaussian_semivariogram(d,s,r,0)
    elif type == 'exponential':
        gamma = exponential_semivariogram(d,s,r,0)
    else:
        'Oops, you selected an incorrect type of semivariogram'
    gamma_inf = np.ones((d.shape))*s
    Q = gamma_inf - gamma
    return Q

# return a stationary random field with mean 0 following a given distribution
def gen_SRF(Q):
    # Decompose the covariance matrix.
    try:
        L = np.linalg.cholesky(Q)
    except:
        try:
#            print('Cholesky decomposition failed. Attempting regularization')
            reg = np.eye(Q.shape[0])*1e-12
            L = np.linalg.cholesky(Q + reg)
        except:
#            print('Regularized Cholesky decomposition failed. Trying SVD instead.')
            U,S,V = np.linalg.svd(Q)
            L = U.dot(np.diag(np.sqrt(S)))

    # Generate pseudorandom numbers
    v = np.random.normal(0,1,Q.shape[0])

    # Compute random field
    w = L.dot(v)

    # Subtract mean
    w = w - np.mean(w)
    return w

# exponential semivariogram
def exponential_semivariogram(h,s,r,a):
    gamma = a+(s-a)*(1.-np.exp(-3*h/r))
    gamma[h==0] = 0
    return gamma

# gaussian semivariogram
def gaussian_semivariogram(h,s,r,a):
    gamma = a + (s-a)*(1.-np.exp(-3*h**2/r**2))
    gamma[h==0] = 0
    return gamma

def covariance(D, scale1, scale2):
    # Given an array of distances and two arrays containing the scales of the observations,
    # compute the semivariance at each pair of locations

    # specify the sill, range and nugget (s,r,a)
    if (scale1 == 1 and scale2 == 10) or (scale1 == 10 and scale2 == 1):
        sm,rm,am = 0.0044, 10000, 0.0022
    elif (scale1 == 30 and scale2 == 10) or (scale1 == 10 and scale2 == 30):
        sm,rm,am = 0.00125, 50000, 0.0004
    elif (scale1 == 30 and scale2 == 1) or (scale1 == 1 and scale2 == 30):
        sm,rm,am = 0.0026, 50000, 0.0019
    elif (scale1 == 1 and scale2 == 1):
        sm,rm,am = 0.0098, 5000, 0.0
    elif (scale1 == 10 and scale2 == 10):
        sm,rm,am = 0.002, 30000, 0.0
    elif (scale1 == 30 and scale2 == 30):
        sm,rm,am = 0.0008, 60000, 0.0
    elif (scale1 == 2 and scale2 == 2):
        sm,rm,am = 0.007, 7000, 0.0
    else:
        print('error')

    if hasattr(D,'__len__'):
        shape = D.shape
    else:
        D = np.array([[D]])
        shape = D.shape

    gamma_m_h = gaussian_semivariogram(D,sm,rm,am)
    gamma_m_inf = np.ones((shape))*sm
    c_h = gamma_m_inf - gamma_m_h
    return c_h

# Determine simple cokriging weights
def ordinary_kriging(x,xi,scale_background,scale_obs,Lx):
    # Solve Aw=c where
    # A: matrix of covariances of background state padded with 1's and a 0 at (N,N)
    # w: kriging weights
    # b: vector of crosscovariances between the observation and the background points

    # Build A
    nx = np.size(x)
    X = np.array([x[0,:],]*nx)
    distance_1 = np.abs(X-X.T)
    distance_2 = np.abs(X+2*Lx-X.T)
    distance_3 = np.abs(X-2*Lx-X.T)
    # D=np.copy(distance_1)
    D = np.minimum(distance_1, distance_2)
    D = np.minimum(D, distance_3)
    A = covariance(D, scale_background, scale_obs)

    # Build c
    distance_1 = np.abs(x-xi)
    distance_2 = np.abs(x+2*Lx-xi)
    distance_3 = np.abs(x-2*Lx-xi)
    D = np.minimum(distance_1, distance_2)
    D = np.minimum(D, distance_3)
    # D=np.copy(distance_1)
    c = covariance(D, scale_background, scale_obs).T

    # Append Lagrange Multiplier
    A = np.append(A,np.ones((np.size(x),1)),axis=1)
    A = np.append(A,np.ones((1,np.size(x)+1)),axis=0)
    A[np.size(x),np.size(x)] = 0
    c = np.append(c,np.array([[1]]),axis=0)

    # Solve equation
    weights = np.linalg.solve(A,c)

    # Remove Lagrange Multipliers
    weights = weights[0:weights.size-1]
    A = A[0:np.sqrt(A.size)-1,0:np.sqrt(A.size)-1]
    c = c[0:c.size-1]
    return weights, A, c

def interpolate(z,x,xi,scale_obs,scale_background,Lx):
    m = np.average(z)
    weights, A, c = ordinary_cokriging(x,xi,scale_background, scale_obs, Lx)
    z_i = weights.T.dot(z.T)
    return z_i, np.squeeze(weights), A, np.squeeze(c)

def plot_B():
    B = np.load('B.npy')
    B = B[0:600,0:600]
    plt.imshow(B)
    plt.clim((-0.05,0.15))
    plt.colorbar()
    plt.show()
