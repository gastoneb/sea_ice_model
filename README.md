# sea_ice_model

This is a two-category sea ice dynamic-thermodynamic model based on Hibler's (1979) viscous-plastic formulation, implementing the implicit solution to the momentum equation described by Lemieux et al. (2008) on a one-dimensional periodic grid. The model is forced mechanically by shallow water models representing the wind and ocean. Ice grows as a function of the thickness, concentration, and a random variable. The model is effectively a random simulation of sea ice dynamics. 

The model comes with a data assimilation class and a few driver programs for conducting twin experiments. The purpose of this framework is to facilitate the evaluation of different data assimilation methods. 

# References:
J.F. Lemieux, B. Tremblay, S. Thomas, J. Sedláček, & L. Mysak, "Using the preconditions Generalized Minimum RESidual (GMRES) method to solve the sea-ice momentum equation," J. of Geophys. Res., 113, 2008.

W.D. Hibler III, "A Dynamic Thermodynamic Sea Ice Model," Journal of Physical Oceanography, 1979.
