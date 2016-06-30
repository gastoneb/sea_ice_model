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
    ocean = Ocean()
    atm = Atmosphere()
    ice_truth = Ice()

    # Read spin-up data as the starting point for the assimilation experiment.
    ice_truth.u = np.load('u.npy')
    ice_truth.h = np.load('h.npy')
    ice_truth.a = np.load('a.npy')*0.9

    # Instantiate the data assimilation class
    oi = OI()
    oi.members.append(Ice())
    oi.x_t[:] = ice_truth.h.T

    # Change some parameters
    oi.members[0].growth_scaling = 0.2
    ice_truth.growth_scaling = 0.2
    ocean.length_scale = 100000
    ocean.time_scaling = 0.25
    atm.length_scale = 50000
    atm.time_scaling = 0.25

    # Instantiate an ice class, stored in OI, as the background state / forecast model
    oi.members.append(Ice())
    oi.members[0].a = np.copy(ice_truth.a)
    oi.members[0].h = np.copy(ice_truth.h)
    oi.x_b = np.copy(oi.members[0].h.T)
    oi.build_H()
    oi.build_R()
    oi.build_B()
    # oi.perturb_state()

    # Some diagnostics you could use later.
    rmse_background = np.sqrt(np.mean((oi.x_b-oi.x_t)**2))
    rmse_analysis = np.sqrt(np.mean((oi.x_a-oi.x_t)**2))

    # Some error covariance matrices for adding incrememntal errors to the thickness
    # and concentration each time step.
    Q_h = gen_covmatrix(oi.members[0].dist,10000,0.0001,"exponential")
    Q_a = gen_covmatrix(oi.members[0].dist,10000,0.00001,"exponential")

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
        if t % atm.dt == 0:
            atm.time_step()
        if t % oi.members[0].dt == 0:
            print('Ice time step at t = '+str(t/3600)+' hours')
            e_h, e_h_t, e_a, e_a_t = gen_SRF(Q_h), gen_SRF(Q_h), gen_SRF(Q_a), gen_SRF(Q_a)
            oi.members[0].h += e_h
            oi.members[0].a += e_a
            ice_truth.h += e_h_t
            ice_truth.a += e_a_t
            oi.members[0].time_step(ocean.u,atm.u)
            ice_truth.time_step(ocean.u,atm.u)
            oi.x_t = np.copy(ice_truth.h.T)
        if t % oi.dt == 0:
            rmse_background = np.sqrt(np.mean((oi.x_b-oi.x_t)**2))
            print("RMSE: "+str(rmse_background))
            oi.x_b = np.copy(oi.members[0].h.T)
            oi.generate_observations()
            oi.analysis()
            oi.members[0].h = np.copy(oi.x_a.T)
            rmse_background = np.sqrt(np.mean((oi.x_b-oi.x_t)**2))
            print("RMSE: "+str(rmse_background))
        if t % tp == 0:
            figure_update_oi(oi.members[0].plot_bool,atm.grid,oi.grid_obs,ocean.u,atm.u,oi.members[0].u, ice_truth.u,oi.members[0].a, ice_truth.a,oi.members[0].h,ice_truth.h,oi.y.T,t)

        if t % (72*3600) == 0:
            print('restart atmosphere', t) # Periodically restart the SW models to prevent oscillations
            atm.restart()
        if t % (120*3600) == 0:
            print('restart ocean', t) # Periodically restart the SW models to prevent oscillations
            ocean.restart()

###############################################################################
# Run the program
###############################################################################
if __name__ == "__main__":
    main()
