# A set of miscellaneous utility/plotting functions.
# Graham Stonebridge
# Department of Systems Design Engineering
# University of Waterloo
# 2015


from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

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

# return minimum distance between a set of points on a periodic 1D grid
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

# return minimum distances between two different vectors of points on a 1D periodic grid
def distance_periodic_2(x1,x2,Lx):
    X1 = np.repeat(x2,x1.size,axis=0)
    X2 = np.repeat(x1,x2.size,axis=0)
    distance_1 = np.abs(X1-X2.T)
    distance_2 = np.abs(X1+2*Lx-X2.T)
    distance_3 = np.abs(X1-2*Lx-X2.T)
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
    elif type == 'linear':
        gamma = linear_semivariogram(d,s,np.amax(d),0)
        print( gamma.T)
    else:
        'Invalid semivariogram selected'
    gamma_inf = np.ones((d.shape))*s
    Q = gamma_inf - gamma
    return Q

# return a stationary random field with mean 0 following a given distribution
def gen_SRF(Q):
    # print(np.linalg.cond(Q))
    # Decompose the covariance matrix.
    try:
        L = np.linalg.cholesky(Q)
    except:
        try:
            # print('Cholesky decomposition failed. Attempting regularization to decrease condition number. ')
            reg = np.eye(Q.shape[0])*1e-12
            L = np.linalg.cholesky(Q + reg)
        except:
            print('Cholesky decomposition failed. Regularization failed. The condition number is likely too high for\
                    conventional approaches. Use the FFT approach instead.')
#            U,S,V = np.linalg.svd(Q)
#            L = U.dot(np.diag(np.sqrt(S)))

    # Generate pseudorandom numbers
    v = np.random.normal(0,1,Q.shape[0])

    # Compute random field
    w = L.dot(v)

    # Subtract mean
    w = w - np.mean(w)
    return w

# Use the fast fourier transform to generate a stationary markov random field. (In progress!)
# Assumes that x goes from -Lx to +Lx-dx.
def gen_srf_fft(x,s,r,shape):
    d = -np.amin(x) + x
    d = - np.amin(x) - np.abs(x)

    l = np.amax(x)*2

    if shape == "gaussian":
        c_x = -gaussian_semivariogram(d,s,r,0) + s
    elif shape == "exponential":
        c_x = -exponential_semivariogram(d,s,r,0) + s
    elif shape == "linear":
        c_x = -linear_semivariogram(d,s,np.amax(d),0) + s
    else:
        print("invalid semivariogram")
    c_x_hat = np.fft.fft(c_x) +0.0j

    a = np.random.normal(0,1,x.size) 
    b = np.random.normal(0,1,x.size)*1.0j
    
    phi = np.real(np.fft.ifft(np.sqrt(np.fft.fft(c_x))*(np.fft.fft(a+b))))
    return phi 

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

def linear_semivariogram(h,s,r,a):
    slope = s/np.amax(np.abs(h))
    gamma = h*slope
    gamma[h==0] = 0
    return gamma

# Return weights for linear interpolation
def interpolate(z,x,xi,scale_obs,scale_background,Lx):
    m = np.average(z)
    weights, A, c = ordinary_cokriging(x,xi,scale_background, scale_obs, Lx)
    z_i = weights.T.dot(z.T)
    return z_i, np.squeeze(weights), A, np.squeeze(c)

# Plotting functions:
def figure_init(plot_bool):
    if plot_bool:
        plt.show(block=False)
        plt.ion()
        plt.figure(figsize=(8,8))

def figure_update(plot_bool,uw,ua,u,a,h,t):
    if plot_bool:
        plt.clf()
        plt.subplot(5,1,1)
        plt.plot(uw.T,'-b',linewidth=2, label='Ocean velocity')
        plt.tick_params(labelbottom="off")
        plt.title('Ocean velocity (m/s)',y=0.7,x=0.85)
        plt.ylim(-0.3,0.3)
        plt.subplot(5,1,2)
        plt.plot(ua.T*150,'-b',linewidth=2, label='Wind Velocity') #multiplied by a constant
        plt.tick_params(labelbottom="off")
        plt.title('Wind velocity (m/s)',y=0.7,x=0.85)
        plt.ylim(-12,12)
        plt.subplot(5,1,3)
        plt.plot(u.T,'-r',linewidth=2, label='Ice velocity')
        plt.tick_params(labelbottom="off")
        plt.title('Ice Velocity (m/s)',y=0.7,x=0.85)
#            plt.ylim([-0.3,0.3])
        plt.subplot(5,1,4)
        plt.plot(h.T,'-r', linewidth=2, label='Ice thickness')
        plt.tick_params(labelbottom="off")
        plt.title('Thickness (m)',y=0.7,x=0.85)
        plt.subplot(5,1,5)
        plt.plot(a.T,'-r', linewidth=2, label='Ice concentration')
        plt.title('Concentration (0-1)',y=0.7,x=0.85)
        plt.show(block=False)
        plt.xlabel('Distance (km)')
        plt.pause(0.0001)
        # plt.savefig('img2/'+str(int(t))+'.png')
        # plt.tight_layout()
        plt.draw()

def figure_update_oi(plot_bool,x,x_obs,uw,uw_t,ua,ua_t,u,u_t,a,a_t,h,h_t,h_obs,t):
    if plot_bool:
        plt.clf()
        plt.subplot(5,1,1)
        plt.plot(x,uw.T,'-r',linewidth=1, label='Ocean velocity')
        plt.plot(x, uw_t.T, '-g', linewidth=1, label = 'Ocean velocity (truth)')
        plt.tick_params(labelbottom="off")
        plt.title('Ocean velocity (m/s)',y=0.7,x=0.85)
        plt.ylim(-0.3,0.3)
        plt.subplot(5,1,2)
        plt.plot(x,ua.T*75,'-r',linewidth=1, label='Wind Velocity') #multiplied by a constant
        plt.plot(x,ua_t.T*75,'-g',linewidth=1, label='Wind Velocity (truth)') #multiplied by a constant
        plt.tick_params(labelbottom="off")
        plt.title('Wind velocity (m/s)',y=0.7,x=0.85)
        plt.ylim(-12,12)
        plt.subplot(5,1,3)
        plt.plot(x,u.T,'-r',linewidth=1, label='Ice velocity')
        plt.plot(x,u_t.T,'-g',linewidth=1, label='Ice velocity (truth)')
        plt.tick_params(labelbottom="off")
        plt.title('Ice Velocity (m/s)',y=0.7,x=0.85)
#            plt.ylim([-0.3,0.3])
        plt.subplot(5,1,4)
        plt.plot(x,h.T,'-r', linewidth=1, label='Ice thickness')
        plt.plot(x,h_t.T,'-g', linewidth=1, label='Ice thickness (truth)')
        plt.plot(x_obs,h_obs.T,'.b', linewidth=1, label='Ice thickness (obs)')
        plt.tick_params(labelbottom="off")
        plt.title('Thickness (m)',y=0.7,x=0.85)
        plt.subplot(5,1,5)
        plt.plot(x,a.T,'-r', linewidth=1, label='Ice concentration')
        plt.plot(x,a_t.T,'-g', linewidth=1, label='Ice concentration (truth)')
        plt.title('Concentration (0-1)',y=0.7,x=0.85)
        plt.show(block=False)
        plt.xlabel('Distance (km)')
        plt.pause(0.0001)
        # plt.savefig('img2/'+str(int(t))+'.png')
        # plt.tight_layout()
        plt.draw()
