# EN.553.738 High-Dimensional Approximations, Probability, and Statistical Learning Final Project

### Charlie Guan, Eric Hu, Jeffrey Zhang @ Johns Hopkins University

In this project, we explore three different nonlinear dimensionality reduction/manifold learning algorithms: Isomap, Locally Linear Embedding (LLE), and Diffusion Map/Laplacian Eigenmap. We benchmark these algos on datasets such as the classic Swiss Roll, Iris, MNIST, and neuronal spikes data. We also compare them to Principal Component Analysis (PCA), which is a linear dimensionality reduction algorithm. Lastly, we have a demo using diffusion map to analyze the free energy landscape of a toy molecular dynamics simulation of a hydrogen dimer. 

To run any of the benchmark/demo, start the corresponding script: 'python filename.py'. You can edit the hyperparameters like the projected dimension and the number of neighbors used in each algorithm inside each script. 

## Prerequisites

The scripts require:

- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [pandas](https://pandas.pydata.org/)
- [rmsd](https://www.cs.cmu.edu/~aarti/Class/10701/slides/Lecture21_1.pdf)
- [pyDiffMap](https://github.com/DiffusionMapsAcademics/pyDiffMap)

## Datasets

- The Swiss Roll and Iris datasets were generated from the 'sklearn.datasets' package. 
- We downloaded the MNIST database from Yann Lecun's webpage [here](http://yann.lecun.com/exdb/mnist/). 
- The neuron spike data came from Adamos DA, Laskaris NA, Kosmidis EK, Theophilidis G. “NASS: an empirical approach to spike sorting with overlap resolution based on a hybrid noise-assisted methodology“. Journal of Neuroscience Methods 2010, vol. 190(1), pp.129-142. |  http://dx.doi.org/10.1016/j.jneumeth.2010.04.018. 
- The molecular dynamics data is from pyDiffMap's [examples](https://github.com/DiffusionMapsAcademics/pyDiffMap/tree/master/examples/Data)

