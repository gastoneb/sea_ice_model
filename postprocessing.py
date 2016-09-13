import numpy as np
import matplotlib.pyplot as plt
from utils import *


# go from a 1 km grid to a 10 or 50 km grid
def change_scale(x,n):
    result = np.zeros([x.shape[0],1000/n])
    for i in range(1,n):
        xi = np.roll(x,i,1)
        result += xi[:,::10]
    result /= 10
    return result

# Load the model state histories
ui = np.load("results/u_hist.npy")
ua = np.load("results/ua_hist.npy")
uo = np.load("results/uw_hist.npy")
h = np.load("results/h_hist.npy")
a = np.load("results/a_hist.npy")

# Plot ui just to see dimensions
#plt.imshow(ui)
#plt.colorbar()
#plt.show()

# Compute the histogram of absolute divergence, du/dx (1/day) on a 10km grid
ui_10km = change_scale(ui,10)
    
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




