#!/usr/bin/env python
# main_ensemble.py
#
# Run a bunch of sea ice models.
#
# Graham Stonebridge
# Department of Systems Design Engineering
# University of Waterloo
# 2015

###############################################################################
# Import Libraries
###############################################################################

import numpy as np
import profile
import sys

# Import custom classes
from world import Model, Ice, Ocean, Atmosphere
from utils import *

###############################################################################
# Main code
###############################################################################
def main():
    # Instantiate sea ice, ocean and atmosphere classes
    ice_restart = Ice()
    ocean = Ocean()
    atm = Atmosphere()
    figure_init(ice_restart.plot_bool)

    # You can define the initial conditions here if you have a saved state
    ice_restart.u = np.load('u.npy')
    ice_restart.h = np.load('h.npy')*0.3
    ice_restart.a = np.load('a.npy')*0.9
    ice_restart.a = 0.85+gen_srf_fft(ice_restart.grid,0.025,100000,'exponential')
    ice_restart.a[ice_restart.a>1] = 1.0
    ice_restart.a[ice_restart.a<0] = 0.1


    ice = []
    n_models = 20
    for i in range(0,n_models):
        ice.append(Ice())
        ice[i].perturb_parameters()
        ice[i].h = np.copy(ice_restart.h)
        ice[i].a = np.copy(ice_restart.a)
        ice[i].u = np.copy(ice_restart.u)

    # Want to save hourly state
    u_hist = np.copy(ice_restart.u)
    ua_hist = np.copy(atm.u)
    uw_hist = np.copy(ocean.u)
    a_hist = np.copy(ice_restart.a)
    h_hist = np.copy(ice_restart.h)

    # Change some parameters
    ocean.length_scale = 50000
    ocean.time_scaling = 0.05
    atm.length_scale = 20000
    atm.time_scaling = 0.1
    tf = 24*3600*30
    ocean.restart()
    atm.restart()

    # March models forward in time
    t = ice[0].t0
    print("Beginning at time "+str(t)+" hours")
    dt = np.amin([ice[0].dt, ocean.dt, atm.dt])
    tp = np.copy(ice[0].dt)*1 # when to update plots
    while True:
        t += dt

        # Check if run is finished
        if t > tf:
            break

        # March forward the relevant models
        if t % ocean.dt == 0:
            ocean.time_step()
        if t % atm.dt == 0:
            atm.time_step()
        if t % ice[0].dt == 0:
            print('Ice time step at t = '+str(t/3600)+' hours')

            ice_mean_h = np.zeros(ice[i].h.shape)
            ice_mean_a = np.zeros(ice[i].h.shape)
            ice_mean_u = np.zeros(ice[i].h.shape)
            for i in range(0,n_models):
                ice[i].time_step(np.copy(ocean.u), np.copy(atm.u))
                ice_mean_h += ice[i].h/n_models
                ice_mean_a += ice[i].a/n_models
                ice_mean_u += ice[i].u/n_models
            u_hist = np.vstack([u_hist,ice_mean_u])
            ua_hist = np.vstack([ua_hist,atm.u])
            uw_hist = np.vstack([uw_hist,ocean.u])
            a_hist = np.vstack([a_hist,ice_mean_a])
            h_hist = np.vstack([h_hist,ice_mean_h])
        if t % tp ==0:
            figure_update(ice[0].plot_bool,ocean.u,atm.u,ice_mean_u,ice_mean_a,ice_mean_h,t)
        if t % (24*3600) == 0:
            growth_scaling = np.random.uniform(-0.2,0.5)
            for i in range(0,n_models):
                ice[i].growth_scaling = growth_scaling + np.random.normal(0,ice[i].growth_err_std)
            print("ice growth scaling set to "+str(growth_scaling))
#        if t % (15*24*3600) == 0:
#            print('restart atmosphere', t) # Periodically restart the SW models to prevent oscillations
#            atm.restart()
#        if t % (15*24*3600) == 0:
#            print('restart ocean', t) # Periodically restart the SW models to prevent oscillations
#            ocean.restart()

    # Save state to file.
    np.save('results/u_mean.npy', ice_mean_u)
    np.save('results/a_mean.npy', ice_mean_a)
    np.save('results/h_mean.npy', ice_mean_h)
    np.save('results/ice.npy', ice)

    np.save('results/u_hist.npy', u_hist)
    np.save('results/uw_hist.npy', uw_hist)
    np.save('results/ua_hist.npy', ua_hist)
    np.save('results/h_hist.npy', h_hist)
    np.save('results/a_hist.npy', a_hist)

###############################################################################
# Run the program
###############################################################################
if __name__ == "__main__":
    main()
