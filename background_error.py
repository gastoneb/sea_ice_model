import numpy as np
import matplotlib.pyplot as plt
from world import Ice

def main():
    ice = np.load("results/ice.npy")
    n_models = len(ice)
    n_x = ice[0].Nx

    # plot hist of thicknesses at one grid point
    point = []
    logpoint = []
    for model in ice:
        point.append(model.h[0,0])
        logpoint.append(np.log(model.h[0,0]))
    point = np.array(point)
    plt.hist(point)
    plt.show()
    plt.hist(logpoint)
    plt.show()


    # Make a B matrix
    X = np.zeros((n_x, n_x))
    x_mean = np.zeros(n_x)
    for i in range(0,n_models):
        x_mean += np.ravel(ice[i].h)/n_models
    for i in range(0,n_models):
        X[:,i] = np.ravel(ice[i].h) - x_mean
    B = X.dot(X.T)/(n_models-1)

    plt.imshow(B)
    plt.colorbar()
    plt.show()

    # Plot the ensemble
    for i in range(0,n_models):
        plt.plot(ice[i].h.T)
    plt.show()

    # Plot error (spread?) vs thickness. 

if __name__ == "__main__":
    main()
