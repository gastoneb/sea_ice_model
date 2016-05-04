# sea_ice_model

A one-dimensional sea ice dynamic-thermodynamic model based on Hibler's (1979) viscous-plastic formulation, implementing the implicit solution to the momentum equation described by Lemieux et al. (2008). The model is forced by a shallow water model. Ice grows as a function of the thickness and concentration.

The model is written in Python 3.5 and allows the user to experiment with different numerical solvers. For the small 200-grid point test domain, direct solution of the system of equations is the fastest.

This model was created partly as an exercise and partly as the basis for some ice thickness data assimilation experiments.

Please send me a message if something appears terribly incorrect. 

# References:
J.F. Lemieux, B. Tremblay, S. Thomas, J. Sedláček, & L. Mysak, "Using the preconditions Generalized Minimum RESidual (GMRES) method to solve the sea-ice momentum equation," J. of Geophys. Res., 113, 2008.

W.D. Hibler III, "A Dynamic Thermodynamic Sea Ice Model," Journal of Physical Oceanography, 1979.
