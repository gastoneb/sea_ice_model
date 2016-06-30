# assimilation.py
# Graham Stonebridge
# Department of Systems Design Engineering
# University of Waterloo
# 2015

###############################################################################
# Import Libraries
###############################################################################

import numpy as np
from world import Model
from utils import *

###############################################################################
# Data Assimilation Class (Optimal Interpolation)
###############################################################################
class OI(Model):
    """A Class for performing Optimal Interpolation"""
    def __init__(self):
        Model.__init__(self)
        # Data assimilation details
        self.dt = 6*3600
        self.n_var = 1
        self.members = []
        self.R_truth = np.zeros((self.Nx, self.Nx))
        self.R_estimate = np.zeros((self.Nx, self.Nx))
        self.error = np.zeros((5,self.n_var+1))
        self.bias =  np.zeros((5,self.n_var+1))
        self.it_statistics = 0
        self.x_b = np.zeros((self.Nx,1))
        self.x_a = np.zeros((self.Nx,1))
        self.x_t = np.zeros((self.Nx,1))
        self.grid_model = self.grid.reshape((self.Nx,1))

        self.background_error_variance = 0.01
        self.background_error_model = 'additive'
        self.background_error_decorrelation_length = 0

        # synthetic sensor details
        self.obs_resolution = 30000
        self.obs_dx = 30000
        self.grid_obs = np.arange(-self.Lx,self.Lx, self.obs_dx)+0.1
        self.n_obs = self.grid_obs.size
        self.grid_obs = self.grid_obs.reshape((self.n_obs,1))
        self.y = np.zeros((self.n_obs,1))
        self.observation_error_variance_truth = 0.001
        self.observation_error_variance_estimate = 0.001
        self.observation_error_model_truth = "additive"
        self.observation_error_model_estimate = "additive"
        self.observation_error_decorrelation_length_truth = 0
        self.observation_error_decorrelation_length_estimate = 0
        self.truth = np.zeros((self.Nx,1))
        self.H_observation = np.zeros((self.n_obs, self.Nx))
        self.H = np.zeros((self.n_obs, self.Nx))
        self.forward_model_method = 'footprint' #"linear interpolation"
        self.observation_method = 'footprint'
        self.dist = distance_periodic(self.grid_obs,self.Lx)
        print('Instantiating OI Class')

    def build_B(self):
        if self.background_error_decorrelation_length == 0:
            self.B = np.eye(self.Nx)*self.background_error_variance
        else:
            D = distance_periodic(self.grid_model,self.Lx)
            self.B = gen_cov

    # Generate the forward model / observation operator
    def build_H(self):
        # Define H for the Kalman Filter
        H = np.zeros((self.n_obs,self.Nx))
        D = distance_periodic_2(self.grid_obs.T, self.grid_model.T, self.Lx)
        if self.forward_model_method == 'linear interpolation':
            for i in range(0,self.n_obs):
                if np.any(D[i,:]) == 0:
                    H[i,D[i,:]==0] = 1
                else:
                    sort_d = np.sort(D[i,:])
                    H[i,D[i,:]==sort_d[0]] = 1-sort_d[0]/(sort_d[0]+sort_d[1])
                    H[i,D[i,:]==sort_d[1]] = 1-sort_d[1]/(sort_d[0]+sort_d[1])
        elif self.forward_model_method == 'footprint':
            for i in range(0,self.n_obs):
                D_col_i = np.copy(D[i,:])
                count = D_col_i[D_col_i<self.obs_resolution/2].size
                H[i,D[i,:]<self.obs_resolution/2] = 1/count
        else:
            print('Invalid forward operator type selected')
        self.H = np.copy(H)
        # Define a possibly different H for generating observations
        H = np.zeros((self.n_obs,self.Nx))
        if self.observation_method == 'linear interpolation':
            for i in range(0,self.n_obs):
                if np.any(D[i,:]) == 0:
                    H[i,D[i,:]==0] = 1
                else:
                    sort_d = np.sort(D[i,:])
                    H[i,D[i,:]==sort_d[0]] = 1-sort_d[0]/(sort_d[0]+sort_d[1])
                    H[i,D[i,:]==sort_d[1]] = 1-sort_d[1]/(sort_d[0]+sort_d[1])
        elif self.observation_method == 'footprint':
            for i in range(0,self.n_obs):
                D_col_i = np.copy(D[i,:])
                count = D_col_i[D_col_i<self.obs_resolution/2].size
                H[i,D[i,:]<self.obs_resolution/2] = 1/count
        else:
            print('Invalid forward operator type selected')
        self.H_observation = np.copy(H)

    # Generate the observation error covariance matrix
    def build_R(self):
        # REcently I've just been using a diagonal R. But you could use the gen_covmatrix function in
        # the utilities file to generate a spatially correlated R.
        if self.observation_error_model_truth == "additive":
            self.R_truth = np.eye((self.n_obs))*self.observation_error_variance_truth
        elif self.observation_error_model_truth == "multiplicative":
            self.R_truth = np.diag(self.H.dot(self.x_t[0:self.Nx])*self.observation_error_variance_truth)
        else:
            print("Invalid observation error model selcted")
        if self.observation_error_model_estimate == "additive":
            self.R_estimate = np.eye((self.n_obs))*self.observation_error_variance_estimate
        elif self.observation_error_model_estimate == "multiplicative":
            self.R_estimate = np.diag(self.H.dot(self.x_b[0:self.Nx])*self.observation_error_variance_estimate)
        else:
            print("Invalid observation error model selcted")

    # Synthesize observations (take truth and perturb it)
    def generate_observations(self):
        x = self.H_observation.dot(self.x_t)
        w = np.reshape(gen_SRF(self.R_truth),(self.n_obs,1))
        self.y[:,0] = x[:,0] + w[:,0]

    # Conduct the analysis
    def analysis(self):
        print('Performing analysis')

        if self.background_error_model == 'additive':
            BHT = self.B.dot(self.H.T)
            HBHT = self.H.dot(self.B).dot(self.H.T)

            # Compute filter weights
            K = BHT.dot(np.linalg.inv(self.R_estimate+HBHT)) # Can get away with this approach because the state is small.

            # Perform data assimilation analysis on each member (just one member for OI)
            self.x_a = self.x_b + K.dot(self.y-self.H.dot(self.x_b))
            self.x_b = np.copy(self.x_a)

        elif self.background_error_model == 'log':
            print('not implemented yet')
        else:
            print("select a different error model")

    # Take the initialized ensemble members and perturb them by adding
    # random correlated noise
    def perturb_state(self):
        print('Perturbing ensemble mean from true initial state')
        perturbation = gen_SRF(self.B).reshape((self.Nx,1))
        if self.background_error_model == "additive":
            self.x_b[:] += perturbation
            self.members[0].h[:] = np.copy(self.x_b.T)
        elif self.background_error_model == "log":
            logh = np.log(np.copy(self.x_b))
            logh += perturbation
            self.x_b = np.exp(logh)
        else:
            print("Invalid background error model selected")
