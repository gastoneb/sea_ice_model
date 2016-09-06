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
    # ice.u = np.load('u.npy')
#    ice.h = np.load('h.npy')
#    ice.a = np.load('a.npy')

    # Change some parameters
    ice.growth_scaling = 1.0 #0.05
    ocean.length_scale = 100000
    ocean.time_scaling = 0.25
    atm.length_scale = 50000
    atm.time_scaling = 0.25

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
        if t % tp ==0:
            figure_update(ice.plot_bool,ocean.u,atm.u,ice.u,ice.a,ice.h,t)
        if t % (72*3600) == 0:
            print('restart atmosphere', t) # Periodically restart the SW models to prevent oscillations
            atm.restart()
        if t % (120*3600) == 0:
            print('restart ocean', t) # Periodically restart the SW models to prevent oscillations
            ocean.restart()

    # Save state to file.
    np.save('u.npy', ice.u)
    np.save('a.npy', ice.a)
    np.save('h.npy', ice.h)

###############################################################################
# Run the program
###############################################################################
if __name__ == "__main__":
    main()
