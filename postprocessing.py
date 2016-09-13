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
def daily_average(x):
    result = np.zeros([30,x.shape[1]])
    for i in range(1,48):
        xi = np.roll(x,i,0)
        xi2 = xi[::48,:]
        result += xi2[:30,:]
    return result/48

# Load the model state histories
ui = np.load("results/u_hist.npy")
ua = np.load("results/ua_hist.npy")
uo = np.load("results/uw_hist.npy")
h = np.load("results/h_hist.npy")
a = np.load("results/a_hist.npy")

# Compute the histogram of absolute divergence, du/dx (1/day) on a 10km grid
ui_10km = change_length_scale(ui,10)
    
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
plt.xlabel("Absolute Divergence Rate (1/day)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Compute temporal autocorrelation of ice velocity
import statsmodels.graphics.tsaplots as tsaplots
daily_10km_ui = daily_average(ui_10km)
plt.imshow(daily_10km_ui)
plt.show()
ui_acf = tsaplots.plot_acf(daily_10km_ui[:,1], lags=20)

plt.show()

