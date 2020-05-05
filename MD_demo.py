import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3

import os

# LOAD DATA 

'''
The Molecular Dynamics simulation shows the trajectory of
two hydrogen particles connected by a double-well potential.
Simulated at 300k with Langevin dynamics in openMM. 

Data taken from https://github.com/DiffusionMapsAcademics/pyDiffMap/

the simulation contains 1000 steps, where each step is a "frame" in the
dimer trajecotry 

energy contains the potential energy of each time frame of the simulation
    --> energy is a vector of length 1000

traj contains the xyz information of each of the two particle at each frame
    --> traj is a (1000, 2, 3) numpy array where first coordinate refers to
        the timestep, second coord refers to which atom, and third coord refers
        to the x, y, z coordinates 

'''

trajFile = os.getcwd() + '/Datasets/MD/dimer_trajectory.npy'
energyFile = os.getcwd() + '/Datasets/MD/dimer_energy.npy'
traj = np.load(trajFile)
energy = np.load(energyFile)

t = np.array(range(1000))

#Computes radius between the two particles of the dimer
def compute_radius(coords):
    return np.linalg.norm(coords[:,0,:]-coords[:,1,:], 2, axis=1)

# VISUALISATION
def plot_radius(traj):
    '''
        Visualizes radius of the dimer as the simulation progresses.
    '''
    fig = plt.figure(figsize=[16,6])
    ax = fig.add_subplot(121)

    radius = compute_radius(traj)
    radii = ax.scatter(range(len(radius)), radius, c=radius, s=20, alpha=0.90, cmap=plt.cm.Spectral)
    cbar = fig.colorbar(radii)
    cbar.set_label('Dimer Radius')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Radius (Angstroms)')

    ax2 = fig.add_subplot(122, projection='3d')
    L=2; i=0;

    ax2.scatter(traj[i,0,0], traj[i,0,1], traj[i,0,2], color='b', s=100)
    ax2.scatter(traj[i,1,0], traj[i,1,1], traj[i,1,2], c='r', s=100)
        
    ax2.set_xlim([-L, L])
    ax2.set_ylim([-L, L])
    ax2.set_zlim([-L, L])

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title("Starting frame of the simulation")


#function for visualizing the trajectory 
def animate_scatters(iteration, data, scatters):
    '''
        @param iteration: current iteration of the simulation
        @param data: list of data positions at each iteration
        @param scatters: list of all the scatters

        @return list of scatters with new coordinates
    '''
    for i in range(data[0].shape[0]):
        scatters[i]._offsets3d = (data[iteration][i, 0:1], data[iteration][i, 1:2], data[iteration][i,2:])
    return scatters

#VISUALIZATION
def animate_trajectory(data, L=2, save=True):
    '''
        animates trajectory
    '''

    fig = plt.figure()
    ax = p3.Axes3D(fig)

    # Initialize scatters
    scatters = [ ax.scatter(data[0][i,0:1], data[0][i,1:2], data[0][i,2:]) for i in range(data[0].shape[0]) ]

    # Number of iterations
    iterations = len(data)

    # Setting the axes properties
    ax.set_xlim3d([-L, L])
    ax.set_xlabel('X')
    ax.set_ylim3d([-L, L])
    ax.set_ylabel('Y')
    ax.set_zlim3d([-L, L])
    ax.set_zlabel('Z')

    # Provide starting angle for the view.
    ax.view_init(25, 10)

    ani = animation.FuncAnimation(fig, animate_scatters, iterations, fargs=(data, scatters),
                                       interval=50, blit=False, repeat=True)

    if save:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
        ani.save('trajectory.mp4', writer=writer)



#uncomment and run the following three lines if you want to plot the 
#dimer radii and animate the trajectory:

#plot_radius(traj)
#plt.savefig("Trajectory_radii.png")
#animate_trajectory(traj)


#clone the pyDiffMap library :
# https://github.com/DiffusionMapsAcademics/pyDiffMap.git
from pydiffmap import diffusion_map as dm

# download from https://pypi.python.org/pypi/rmsd/1.2.5
# via pip (pip install rmsd)
import rmsd

#custom metric to construct adjacency matrix for MD data
def RMSDmetric(structure1, structure2):
    numParticles = len(structure1) / 3

    coords1 = structure1.reshape(int(numParticles), 3)
    coords2 = structure2.reshape(int(numParticles), 3)

    coords1 = coords1 - rmsd.centroid(coords1)
    coords2 = coords2 - rmsd.centroid(coords2)

    return rmsd.kabsch_rmsd(coords1, coords2)


data = traj # keep for later
traj = traj.reshape(traj.shape[0], traj.shape[1]*traj.shape[2])
mydmap = dm.DiffusionMap.from_sklearn(n_evecs=1, epsilon=0.05, alpha=0.5, k=1000, metric = RMSDmetric)
dmap = mydmap.fit_transform(traj)

evecs = mydmap.evecs

fig = plt.figure(figsize=[16,6])
ax = fig.add_subplot(121)

ax.scatter(compute_radius(data), evecs[:,0], c=evecs[:,0], s=10, cmap=plt.cm.Spectral)
ax.set_xlabel('Radius')
ax.set_ylabel('Dominant eigenvector')

ax2 = fig.add_subplot(122)
#
cax2 = ax2.scatter(compute_radius(data), energy, c=evecs[:,0], s=10, cmap=plt.cm.Spectral)
ax2.set_xlabel('Radius')
ax2.set_ylabel('Potential Energy')
cbar = fig.colorbar(cax2)
cbar.set_label('Dominant eigenvector')
plt.savefig("MD_diffusion.png")
plt.show()