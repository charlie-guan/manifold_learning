import numpy as np
import matplotlib.pyplot as plt
import algorithms.Isomap as Isomap
import algorithms.Laplacian_Eigenmaps as Laplacian_Eigenmaps
import algorithms.LLE as LLE
import algorithms.PCA as PCA

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

import os

# LOAD DATA (680 data points, 180 features)
file = os.getcwd() + '/Datasets/spikes.csv'
X = np.genfromtxt(file, delimiter=',')
zeros = np.zeros(300)
ones = np.ones(300)
twos = np.ones(50)*2
threes = np.ones(30)*3
Y = np.concatenate((zeros, ones, twos, threes))
Y = [int(temp) for temp in Y]

# VISUALISATION
def plot_original_spikes(X):
    '''
        Plots the original Swiss roll dataset.
    '''
    spikes = X
    xs = range(0,180)

    fig = plt.figure()
    ax = fig.gca()

    for i in range(0, 300):
        if i==0:
            ax.plot(xs, spikes[i,:], c='b', linewidth=0.3, label='Neuron 1')
        else:
            ax.plot(xs, spikes[i,:], c='b', linewidth=0.3)
        

    for i in range(300, 600):
        if i==300:
            ax.plot(xs, spikes[i,:], c='r', linewidth=0.3, label='Neuron 2')
        else:
            ax.plot(xs, spikes[i,:], c='r', linewidth=0.3)

    for i in range(601, 650):
        if i==601:
            ax.plot(xs, spikes[i,:], c='g', linewidth=0.3, label='Overlap of 1 & 2')
        else:
            ax.plot(xs, spikes[i,:], c='g', linewidth=0.3)


    for i in range(650, 680):
        if i==650:
            ax.plot(xs, spikes[i,:], c='y', linewidth=0.3, label='Neuron 3')
        else:
            ax.plot(xs, spikes[i,:], c='y', linewidth=0.3)

    ax.set_title("Original neuron spike dataset")
    leg = plt.legend(loc='lower left')

# VISUALISATION
def plot_low_dimensional_embedding(X_low, name="", time=0, error=0):
    '''
        Plots the low dimensional embedding given by the matrix X_low (n_samples, n_features).

        X_low: The low-dimensional representation.
        name: The (string) name of the technique used for the low-dimensional embedding.
        time: Time the execution took.
        error: The error of the representation.
    '''

    colors = ['b', 'r', 'g', 'y']

    X_low = (X_low - np.min(X_low, 0)) / ( np.max(X_low, 0) -  np.min(X_low, 0))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    
    for i in range(X_low.shape[0]):
        ax.scatter(X_low[i, 0], X_low[i, 1], X_low[i, 2],
                 color='k'
                 )
    '''
    for i in range(0, 600):
        ax.scatter(X_low[i, 0], X_low[i, 1], X_low[i, 2], 
                 color=colors[Y[i]],
                 )

    for i in range(650, 680):
        ax.scatter(X_low[i, 0], X_low[i, 1], X_low[i, 2], 
                 color=colors[Y[i]],
                 )
    '''


    if error != 0:
        plt.title("{}, time: {:.3f}s, error: {:.3e}".format(name, time, error))
    else:
        plt.title("{}, time: {:.3f}s".format(name, time))


        
plot_original_spikes(X)
plt.savefig("original_spikes.png")


num_neighbors = 100
num_components = 3
# PCA
X_pca, tpca = PCA.fit_transform(X, components=num_components)
plot_low_dimensional_embedding(X_pca, "PCA", tpca)

plt.savefig("neuron_pca_nocolors.png")
# LLE
X_lle, tlle, err_lle = LLE.fit_transform(X, n_neighbors=num_neighbors, components=num_components)
plot_low_dimensional_embedding(X_lle, "LLE", tlle, err_lle)
plt.savefig("neuron_lle_nocolors.png")
# Isomap
X_isomap, tisomap, err_isomap = Isomap.fit_transform(X, n_neighbors=num_neighbors, components=num_components)
plot_low_dimensional_embedding(X_isomap, "Isomap", tisomap, err_isomap)
plt.savefig("neuron_isomap_nocolors.png")
# Laplacian Eigenmaps
X_laplacian, tlaplacian = Laplacian_Eigenmaps.fit_transform(X, n_neighbors=num_neighbors, components=num_components)
plot_low_dimensional_embedding(X_laplacian, "Laplacian Eigenmaps", tlaplacian)
plt.savefig("neuron_diffusion_nocolors.png")


plt.show();