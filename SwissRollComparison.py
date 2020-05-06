import numpy as np
import matplotlib.pyplot as plt
import algorithms.Isomap as Isomap
import algorithms.Laplacian_Eigenmaps as Laplacian_Eigenmaps
import algorithms.LLE as LLE
import algorithms.PCA as PCA

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from Datasets import SwissRollLoader

# LOAD DATA
X, colors = SwissRollLoader.load_swissroll(n_datapoints=1000)

# VISUALISATION
def plot_original_swissroll():
    '''
        Plots the original Swiss roll dataset.
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0], X[:,1] ,X[:,2], c=colors, cmap=plt.cm.Spectral)
    ax.set_title("Original Swiss Roll dataset")

# VISUALISATION
def plot_embedding(X_low, name="", time=0, error=0):
    '''
        Plots the low dimensional embedding given by the matrix X_low (n_samples, n_features).

        X_low: The low-dimensional representation.
        name: The (string) name of the technique used for the low-dimensional embedding.
        time: Time the execution took.
        error: The error of the representation.
    '''
    plt.figure()
    ax = plt.subplot(111) 
    ax.scatter(X_low[:, 0], X_low[:, 1], c=colors, s=20, cmap=plt.cm.Spectral)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.axis('tight')
    if error != 0:
        plt.title("{}, time: {:.3f}s, error: {:.3e}".format(name, time, error))
    else:
        plt.title("{}, time: {:.3f}s".format(name, time))

# PCA
X_pca, tpca = PCA.fit_transform(X)
plot_embedding(X_pca, "PCA", tpca)
plt.savefig("swiss_pca.png")
# LLE
X_lle, tlle, err_lle = LLE.fit_transform(X, n_neighbors=100)
plot_embedding(X_lle, "LLE", tlle, err_lle)
plt.savefig("swiss_lle_100neighbors.png")
# Isomap
X_isomap, tisomap, err_isomap = Isomap.fit_transform(X, n_neighbors=100)
plot_embedding(X_isomap, "Isomap", tisomap, err_isomap)
plt.savefig("swiss_isomap_100neighbors.png")
# Laplacian Eigenmaps
X_laplacian, tlaplacian = Laplacian_Eigenmaps.fit_transform(X, n_neighbors=100)
plot_embedding(X_laplacian, "Laplacian Eigenmaps", tlaplacian)
plt.savefig("swiss_diffusion_100neighbors.png")

plot_original_swissroll()
plt.savefig("originalswissroll.png")
plt.show();