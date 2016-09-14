#!/usr/bin/env python
# main.py
#
# Run the ice model.
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
    ice = Ice()
    ocean = Ocean()
    atm = Atmosphere()
    figure_init(ice.plot_bool)

    # You can define the initial conditions here if you have a saved state
    ice.u = np.load('results/u.npy')
    ice.h = np.load('results/h.npy')
    ice.a = np.load('results/a.npy')

    # Want to save hourly state
    u_hist = np.copy(ice.u)
    ua_hist = np.copy(atm.u)
    uw_hist = np.copy(ocean.u)
    a_hist = np.copy(ice.a)
    h_hist = np.copy(ice.h)


    # Change some parameters
    ice.growth_scaling = 0.5
    ocean.length_scale = 100000
    ocean.time_scaling = 0.1
    atm.length_scale = 50000
    atm.time_scaling = 0.1

    # March models forward in time
    t = ice.t0
    print("Beginning at time "+str(t)+" hours")
    dt = np.amin([ice.dt, ocean.dt, atm.dt])
    tp = np.copy(ice.dt)*1 # when to update plots
    while True:
        t += dt

        # Check if run is finished
        if t > ice.tf:
            break

        # March forward the relevant models
        if t % ocean.dt == 0:
            ocean.time_step()
        if t % atm.dt == 0:
            atm.time_step()
        if t % ice.dt == 0:
            print('Ice time step at t = '+str(t/3600)+' hours')
            ice.time_step(ocean.u,atm.u)
            u_hist = np.vstack([u_hist,ice.u])
            ua_hist = np.vstack([ua_hist,atm.u])
            uw_hist = np.vstack([uw_hist,ocean.u])
            a_hist = np.vstack([a_hist,ice.a])
            h_hist = np.vstack([h_hist,ice.h])
        if t % tp ==0:
            figure_update(ice.plot_bool,ocean.u,atm.u,ice.u,ice.a,ice.h,t)
        if t % (24*3600) == 0:
#             ice.growth_scaling = np.random.uniform(-0.2,0.5)
             ice.growth_scaling = np.random.uniform(0.5,0.9)
             print("ice growth scaling set to "+str(ice.growth_scaling))
#        if t % (15*24*3600) == 0:
#            print('restart atmosphere', t) # Periodically restart the SW models to prevent oscillations
#            atm.restart()
#        if t % (15*24*3600) == 0:
#            print('restart ocean', t) # Periodically restart the SW models to prevent oscillations
#            ocean.restart()

    # Save state to file.
    np.save('u.npy', ice.u)
    np.save('a.npy', ice.a)
    np.save('h.npy', ice.h)
#    np.save('results/u_hist.npy', u_hist)
#    np.save('results/uw_hist.npy', uw_hist)
#    np.save('results/ua_hist.npy', ua_hist)
#    np.save('results/a_hist.npy', a_hist)
#    np.save('results/h_hist.npy', h_hist)


###############################################################################
# Run the program
###############################################################################
if __name__ == "__main__":
    main()
