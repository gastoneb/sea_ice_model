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
#def change_time_scale(x,n_half_hrs):
def change_time_scale(x,delt, delt_new)   :
    
    n = int(delt_new/delt)
    p = 30
    result = np.zeros([p,x.shape[1]])
    for i in range(1,n):
        xi = np.roll(x,i,0)
        xi2 = xi[::n,:]
        result += xi2[:p,:]
    return result/n

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}

plt.rc('font', **font)

# Load the model state histories
ui = np.load("results/u_hist.npy")
ua = np.load("results/ua_hist.npy")
uo = np.load("results/uw_hist.npy")
h = np.load("results/h_hist.npy")
a = np.load("results/a_hist.npy")


# Histogram of ice thickness
plt.hist(np.ravel(h),10,normed=True)
plt.ylabel("Frequency")
plt.xlabel("Thickness (m)")
plt.show()

# Plot the state at an intermediate timestep. 
t=1000
plt.subplot(5,1,1)
plt.plot(uo[t,:].T,'-b',linewidth=2, label='Ocean velocity')
plt.tick_params(labelbottom="off")
plt.subplot(5,1,1).set_title('Ocean velocity (m/s)')#,y=0.69,x=0.8,backgroundcolor="white")
plt.ylim(-0.3,0.3)
plt.subplot(5,1,2).set_title("Wind velocity (m/s)")
plt.plot(ua[t,:].T*75,'-b',linewidth=2, label='Wind Velocity') #multiplied by a constant
plt.tick_params(labelbottom="off")
#plt.title('Wind velocity (m/s)',y=0.7,x=0.8)
plt.ylim(-12,12)
plt.subplot(5,1,3).set_title('Ice velocity (m/s)')
plt.plot(ui[t,:].T,'-r',linewidth=2, label='Ice velocity')
plt.tick_params(labelbottom="off")
#plt.ylim([-0.025,0.025])
plt.subplot(5,1,4).set_title("Thickness (m)")
plt.plot(h[t,:].T,'-r', linewidth=2, label='Ice thickness')
plt.tick_params(labelbottom="off")
plt.subplot(5,1,5).set_title("Ice concentration (0-1)")
plt.plot(a[t,:].T,'-r', linewidth=2, label='Ice concentration')
plt.show(block=False)
plt.xlabel('Distance (km)')
plt.tight_layout()
plt.show()


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
hist = np.array(hist)*1.0
for i in range(0,len(hist)):
    hist[i] = float(hist[i])/len(dudx)

plt.scatter(bin_centres,hist)
plt.plot(bin_centres,hist)
plt.xscale('log')
plt.yscale('log')
plt.xlim([10**(-2),10])
plt.ylim([10**(-5),10**0])
plt.xlabel("Divergence Rate (1/day)")
plt.ylabel("Frequency")
plt.grid(which='major')
plt.tight_layout()
plt.show()


# diverence rate plot but not on a log-scale
nbins=600
[hist, bin_edges] = np.histogram(dudx,nbins)
hist = hist/dudx.size
bin_centres = 0.5*(bin_edges[0:nbins]+bin_edges[1:nbins+1])
abs_dudx = np.ravel(np.abs((ui_10km-np.roll(ui_10km,1,axis=1))*3.6*24))
plt.scatter(bin_centres,hist)
plt.plot(bin_centres,hist)
plt.yscale('log')
plt.xlim((-0.4,0.4))
plt.xlabel("Divergence Rate (1/day)")
plt.ylabel("Frequency")
plt.grid(which='major')
plt.tight_layout()
plt.show()
nbins=50

# Compute temporal autocorrelation of ice velocity
import statsmodels.graphics.tsaplots as tsaplots
daily_10km_ui = change_time_scale(ui_10km,0.5,24)
#ui_acf = tsaplots.plot_acf(daily_10km_ui[:,1], lags=20)
import statsmodels.tsa.stattools as stattools
acf=stattools.acf(daily_10km_ui[:,0])
acf = acf*0
for i in range(0,100):
    acf_i = stattools.acf(daily_10km_ui[:,i])
    acf += acf_i/100
plt.clf()
plt.plot(acf)
plt.scatter(np.arange(acf.size),acf)
plt.axhline(y=0, color='black')
plt.xlabel("Time lag (days)")
plt.ylabel("Autocorrelation")
plt.xlim([0,30])
plt.ylim([-0.4,1.0])
plt.title("")
plt.grid(which="major")
plt.tight_layout()
plt.show()

# Compute histogram of wind and ocean current velocities
ocean = np.ravel(uo)
[hist,bin_edges] = np.histogram(ocean,nbins)
bin_centres = 0.5*(bin_edges[0:nbins]+bin_edges[1:nbins+1])
plt.plot(bin_centres,hist/np.sum(hist))
plt.yscale('log')
plt.ylim([0.0001,0.1])
plt.xlim([-0.3,0.3])
plt.xlabel("Ocean velocity (m/s)")
plt.ylabel("Frequency")
plt.grid(which="major")
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
plt.grid(which="major")
plt.show()


# Compute histogram of ice velocity fluctuations, u/stdev(u)
stdev_u = ui_10km.std(axis=0)
fluctuation = np.copy(ui_10km)
#for i in range(0,fluctuation.shape[1]):
#    fluctuation[:,i] /= stdev_u[i]

fluctuation = np.ravel(fluctuation)
[hist,bin_edges] = np.histogram(fluctuation,nbins)
bin_centres = 0.5*(bin_edges[0:nbins]+bin_edges[1:nbins+1])
plt.plot(bin_centres,hist/np.sum(hist))
plt.yscale('log')
plt.ylim([0.00001,1])
#plt.xlim([-6,6])
plt.xlabel("Sea ice velocity fluctuation (m/s)")
plt.ylabel("Frequency")
plt.grid(which="major")
plt.show()



# Create spatial ACF of ice thickness
#tsaplots.plot_acf(h[h.shape[0]-1,:],lags=300)
acf = stattools.acf(h[0,:],nlags=300)*0
for i in range(0,1300):
    acf_i = stattools.acf(h[i,:],nlags=300)
    acf += acf_i/1300
plt.plot(acf)
plt.xlabel("Distance Lag (km)")
plt.ylabel("Autocorrelation")
plt.grid(which="major")
plt.xlim([0,300])
plt.axhline(y=0, color='black')
plt.ylim([-0.3,1])
plt.show()



