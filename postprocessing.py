import numpy as np
import matplotlib.pyplot as plt
from utils import *


# go from a 1 km grid to a 10 or 50 km grid
def change_length_scale(x,n):
    result = np.zeros([x.shape[0],int(int(1000)/int(n))])
    for i in range(1,n):
        xi = np.roll(x,i,1)
        result += xi[:,::10]
    result /= 10
    return result

# go from 30 minute timesteps to  daily averages. Only for 30 days.
def change_time_scale(x,n_half_hrs):
    n = n_half_hrs*2
    p = int(30*24*2/n)
    result = np.zeros([p,x.shape[1]])
    for i in range(1,n):
        xi = np.roll(x,i,0)
        xi2 = xi[::n,:]
        result += xi2[:p,:]
    return result/n

# Load the model state histories
ui = np.load("results/u_hist.npy")
ua = np.load("results/ua_hist.npy")
uo = np.load("results/uw_hist.npy")
h = np.load("results/h_hist.npy")
a = np.load("results/a_hist.npy")

# Compute the histogram of absolute divergence, du/dx (1/day) on a 10km grid
nbins=50
ui_10km = change_length_scale(ui,10)
dudx = np.ravel((ui_10km-np.roll(ui_10km,1,axis=1)))*3.6*24
[hist, bin_edges] = np.histogram(dudx,nbins)
hist = hist/dudx.size
bin_centres = 0.5*(bin_edges[0:nbins]+bin_edges[1:nbins+1])

abs_dudx = np.ravel(np.abs((ui_10km-np.roll(ui_10km,1,axis=1))*3.6*24))
log_abs_dudx = np.log10(abs_dudx)
nbins=50
[hist, bin_edges] = np.histogram(log_abs_dudx,nbins)
log_bin_centres = 0.5*(bin_edges[0:nbins]+bin_edges[1:nbins+1])
bin_centres = 10**(log_bin_centres)
plt.scatter(bin_centres,hist)
plt.plot(bin_centres,hist)
plt.xscale('log')
plt.yscale('log')
plt.xlim([10**(-2),10])
plt.xlabel("Divergence Rate (1/day)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Compute temporal autocorrelation of ice velocity
#import statsmodels.graphics.tsaplots as tsaplots
daily_10km_ui = change_time_scale(ui_10km,24)
#ui_acf = tsaplots.plot_acf(daily_10km_ui[:,1], lags=20)
import statsmodels.tsa.stattools as stattools
acf=stattools.acf(daily_10km_ui[:,0])
acf = acf*0
for i in range(0,100):
    acf_i = stattools.acf(daily_10km_ui[:,i])
    acf += acf_i/100
plt.plot(acf)
plt.scatter(np.arange(acf.size),acf)
plt.axhline(y=0, color='black')
plt.xlabel("Time lag (days)")
plt.ylabel("Autocorrelation")
plt.title("")
plt.show()

# Compute histogram of wind and ocean current velocities
wind = np.ravel(ua*150)
[hist,bin_edges] = np.histogram(wind,nbins)
bin_centres = 0.5*(bin_edges[0:nbins]+bin_edges[1:nbins+1])
plt.plot(bin_centres,hist/np.sum(hist))
plt.yscale('log')
plt.ylim([0.0001,0.1])
plt.xlabel("Wind velocity (m/s)")
plt.ylabel("Frequency")
plt.show()


# Compute histogram of ice velocity fluctuations, u/stdev(u)
stdev_u = ui_10km.std(axis=0)
fluctuation = np.copy(ui_10km)
for i in range(0,fluctuation.shape[1]):
    fluctuation[:,i] /= stdev_u[i]

fluctuation = np.ravel(fluctuation)
[hist,bin_edges] = np.histogram(fluctuation,nbins)
bin_centres = 0.5*(bin_edges[0:nbins]+bin_edges[1:nbins+1])
plt.plot(bin_centres,hist/np.sum(hist))
plt.yscale('log')
plt.ylim([1/1e4,1])
plt.xlabel("u/std(u)")
plt.ylabel("Frequency")
plt.show()



# Create spatial ACF of ice thickness
tsaplots.plot_acf(h[h.shape[0]-1,:],lags=300)
plt.xlabel("Distance Lag (km)")
plt.ylabel("Autocorrelation")
plt.show()
