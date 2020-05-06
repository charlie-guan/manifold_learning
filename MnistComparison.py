import numpy as np
import matplotlib.pyplot as plt
import algorithms.Isomap as Isomap
import algorithms.Laplacian_Eigenmaps as Laplacian_Eigenmaps
import algorithms.LLE as LLE
import algorithms.PCA as PCA

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from Datasets import MnistLoader
from sklearn import datasets

# LOAD DATA
X, Y = MnistLoader.load_mnist(n_samples=5000)


# VISUALISATION
def plot_embedding(X_low, name="", time=0, error=0):
    '''
        Plots the low dimensional embedding given by the matrix X_low (n_samples, n_features).

        X_low: The low-dimensional representation.
        name: The (string) name of the technique used for the low-dimensional embedding.
        time: Time the execution took.
        error: The error of the representation.
    '''
    X_low = (X_low - np.min(X_low, 0)) / ( np.max(X_low, 0) -  np.min(X_low, 0))
    X_low = X_low[:300, :] #extract only the first 300 points for plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(X_low.shape[0]):
        ax.text(X_low[i, 0], X_low[i, 1], X_low[i, 2], str(Y[i]),
                 color=plt.cm.Set1(Y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 8})

    if error != 0:
        plt.title("{}, time: {:.3f}s, error: {:.3e}".format(name, time, error))
    else:
        plt.title("{}, time: {:.3f}s".format(name, time))


        

num_neighbors = 5
num_components = 3
# PCA
X_pca, tpca = PCA.fit_transform(X, components=num_components)
plot_embedding(X_pca, "PCA", tpca)

plt.savefig("mnist_pca_5nbr.png")
# LLE
X_lle, tlle, err_lle = LLE.fit_transform(X, n_neighbors=num_neighbors, components=num_components)
plot_embedding(X_lle, "LLE", tlle, err_lle)
plt.savefig("mnist_lle_5nbr.png")
# Isomap
X_isomap, tisomap, err_isomap = Isomap.fit_transform(X, n_neighbors=num_neighbors, components=num_components)
plot_embedding(X_isomap, "Isomap", tisomap, err_isomap)
plt.savefig("mnist_isomap_5nbr.png")
# Laplacian Eigenmaps
X_laplacian, tlaplacian = Laplacian_Eigenmaps.fit_transform(X, n_neighbors=num_neighbors, components=num_components)
plot_embedding(X_laplacian, "Laplacian Eigenmaps", tlaplacian)
plt.savefig("mnist_diffusion_5nbr.png")


plt.show();