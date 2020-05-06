import numpy as np
import matplotlib.pyplot as plt
import algorithms.Isomap as Isomap
import algorithms.Laplacian_Eigenmaps as Laplacian_Eigenmaps
import algorithms.LLE as LLE
import algorithms.PCA as PCA

from Datasets import IrisLoader
from matplotlib.ticker import NullFormatter

# LOAD DATA
X, Y = IrisLoader.load_iris()

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
    ax.scatter(X_low[:, 0], X_low[:, 1], c=Y, s=20, cmap=plt.cm.rainbow)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.axis('tight')
    if error != 0:
        plt.title("{}, time: {:.3f}s, error: {:.3e}".format(name, time, error))
    else:
        plt.title("{}, time: {:.3f}s".format(name, time))

num_neighbors = 50
# PCA
X_pca, tpca = PCA.fit_transform(X)
plot_embedding(X_pca, "PCA", tpca)
plt.savefig("iris_pca.png")
# LLE
X_lle, tlle, err_lle = LLE.fit_transform(X, n_neighbors=20)
plot_embedding(X_lle, "LLE", tlle, err_lle)
plt.savefig("iris_lle.png")

# Isomap
X_isomap, tisomap, err_isomap = Isomap.fit_transform(X, n_neighbors=50)
plot_embedding(X_isomap, "Isomap", tisomap, err_isomap)
plt.savefig("iris_isomap.png")

# Laplacian Eigenmaps
X_laplacian, tlaplacian = Laplacian_Eigenmaps.fit_transform(X, n_neighbors=10)
plot_embedding(X_laplacian, "Laplacian Eigenmaps", tlaplacian)
plt.savefig("iris_diffusion.png")



plt.show();