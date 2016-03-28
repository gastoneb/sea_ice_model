#!/usr/bin/env python
# main.py
#
# Main code for running the model.
# Python 3.5
#
# Graham Stonebridge
# Department of Systems Design Engineering
# University of Waterloo
# 2015

###############################################################################
# Import Libraries
###############################################################################

import numpy as np

# Import custom classes
from world import World
from world import Ice
from world import Ocean
from world import Atmosphere

###############################################################################
# Main code
###############################################################################
def main():
    # Instantiate a class that describes global model properties.
    model = World('general_input.txt')

    # Instantiate sea ice, ocean and atmosphere classes
    ice = Ice(model)
    ocean = Ocean(model)
    atm = Atmosphere(model)

    # March models forward in time
    t = model.t0
    print("Beginning at time "+str(t)+" hours")
    dt = np.amin([ice.dt, ocean.dt, atm.dt])
    tp = np.copy(ice.dt)*1
    while True:
        t += dt

        # Check if model run is finished
        if t > model.tf:
            break

        # Advance model
        if t % ocean.dt == 0:
            ocean.time_step()
        if t % ice.dt == 0:
            print('Ice time step at t = '+str(t/3600)+' hours')
            ice.time_step(ocean.u,ocean.v)
        if t % tp ==0:
            model.figure_update(ocean.u,ice.u,ice.a,ice.h)
        if t % (50*3600) == 0:
            print('restarting SW model', t) #Ends up with too many high-frequency components otherwise
            ocean.restart()
    np.save('u.npy', ice.u)
    np.save('a.npy', ice.a)
    np.save('h.npy', ice.h)

###############################################################################
# Run the program
###############################################################################
if __name__ == "__main__":
    main()
