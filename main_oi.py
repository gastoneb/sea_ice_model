# main_oi.py
#
# Perform optimal interpolation on a numerical sea ice model using simulated observations.
#
# Graham Stonebridge
# Department of Systems Design Engineering
# University of Waterloo
# 2015

###############################################################################
# Import Libraries
###############################################################################

from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt

# Import custom classes
from world import Model
from world import Ice
from world import Ocean
from world import Atmosphere
from assimilation import OI
from utils import *

###############################################################################
# Main code
###############################################################################
def main():
    # Instantiate model classes
    ocean_truth = Ocean()
    ocean = Ocean()
    atm_truth = Atmosphere()
    atm = Atmosphere()
    ice_truth = Ice()

    # Read spin-up data as the starting point for the assimilation experiment.
#    ice_truth.u = np.load('u.npy')
#    ice_truth.h = np.load('h.npy')
#    ice_truth.a = np.load('a.npy')*0.9

    # Instantiate the data assimilation class
    oi = OI()
    oi.members.append(Ice())
    oi.x_t[:] = ice_truth.h.T

    # Change some parameters
    oi.members[0].growth_scaling = 0.2
    ice_truth.growth_scaling = 0.2
    ocean.length_scale = 100000
    ocean_truth.length_scale = 100000
    ocean.time_scaling = 0.25
    ocean_truth.time_scaling = 0.25
    atm.length_scale = 50000
    atm_truth.length_scale = 50000
    atm.time_scaling = 0.25
    atm_truth.time_scaling = 0.25
    ocean.eta = np.copy(ocean_truth.eta)
    atm.eta = np.copy(atm_truth.eta)

    atm.eta += gen_srf_fft(atm.grid,atm.eta_error_variance,atm.length_scale,'gaussian')
    ocean.eta += gen_srf_fft(ocean.grid,ocean.eta_error_variance,ocean.length_scale,'gaussian')

    # Instantiate an ice class, stored in OI, as the background state / forecast model
    oi.members.append(Ice())
    oi.members[0].a = np.copy(ice_truth.a)
    oi.members[0].h = np.copy(ice_truth.h)
    oi.x_b = np.copy(oi.members[0].h.T)
    oi.build_H()
    oi.build_R()
    oi.build_B()
    oi.perturb_state()
    oi.members[0].h = np.copy(oi.x_b.T)

    # Some diagnostics you could use later.
    rmse_background = np.sqrt(np.mean((oi.x_b-oi.x_t)**2))
    rmse_analysis = np.sqrt(np.mean((oi.x_a-oi.x_t)**2))

    # Set up the plots
    figure_init(ice_truth.plot_bool)

    # March models forward in time
    t = oi.members[0].t0
    print("Beginning at time "+str(t)+" hours")
    dt = np.amin([oi.members[0].dt, ocean.dt, atm.dt])
    tp = np.copy(oi.members[0].dt)*1 # when to update plots
    while True:
        t += dt

        # Check if run is finished
        if t > oi.members[0].tf:
            break

        # March models forward in time
        if t % ocean.dt == 0:
            ocean.time_step()
            ocean_truth.time_step()
        if t % atm.dt == 0:
            atm.time_step()
            atm_truth.time_step()
        if t % oi.members[0].dt == 0:
            print('Ice time step at t = '+str(t/3600)+' hours')
            oi.members[0].time_step(ocean.u, atm.u)
            ice_truth.time_step(ocean_truth.u, atm_truth.u)
            oi.x_t = np.copy(ice_truth.h.T)
        if t % oi.dt == 0:
            rmse_background = np.sqrt(np.mean((oi.x_b - oi.x_t)**2))
            print("RMSE: "+str(rmse_background))
            oi.x_b = np.copy(oi.members[0].h.T)
            oi.generate_observations()
            oi.analysis()
            oi.members[0].h = np.copy(oi.x_a.T)
            rmse_background = np.sqrt(np.mean((oi.x_b - oi.x_t)**2))
            print("RMSE: "+str(rmse_background))
        if t % tp == 0:
            figure_update_oi(oi.members[0].plot_bool,atm.grid,oi.grid_obs,ocean.u,ocean_truth.u,atm.u,atm_truth.u,\
                    oi.members[0].u, ice_truth.u,oi.members[0].a, ice_truth.a,oi.members[0].h,ice_truth.h,oi.y.T,t)
        if t % (72*3600) == 0:
            print('restart atmosphere', t) # Periodically restart the SW models to prevent oscillations
            atm.restart()
            atm_truth.restart()
            atm.eta = np.copy(atm_truth.eta)
            atm.eta += gen_srf_fft(atm.grid,atm.eta_error_variance,atm.length_scale,'gaussian')
        if t % (120*3600) == 0:
            print('restart ocean', t) # Periodically restart the SW models to prevent oscillations
            ocean.restart()
            ocean_truth.restart()
            ocean.eta = np.copy(ocean_truth.eta)
            ocean.eta += gen_srf_fft(ocean.grid,ocean.eta_error_variance,ocean.length_scale,'gaussian')

###############################################################################
# Run the program
###############################################################################
if __name__ == "__main__":
    main()
