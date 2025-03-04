import numpy as np
import matplotlib.pyplot as plt
import torch
import pyvista as pv

## Plotting the topology as a graph

# Remove legend duplicates
def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

# Given encoded data plot graph
def PlotGraph(data,limits=None,manual_axes=False,plot_lines=True):
    fig = plt.figure(figsize=(6, 20))
    ax = fig.add_subplot(111, projection='3d')
    # For every contact in topology plot line
    if plot_lines == True:
        for line_idx in data.edge_index.T:

            # Define radius for normalized x y z coordinates
            radius = data.x[0,0]   
            #radius = 1                                                     
            plotx, ploty, plotz = [data.pos[line_idx][:,i]/radius for i in [0,1,2]]

            # Different colours for P-W and P-P contact
            if np.any(data.mask[line_idx]==False):
                clr = 'indianred'
                lbl = 'P-W Contact'
            else:
                clr = 'steelblue'
                lbl = 'P-P Contact'
            ax.plot(plotx,ploty,plotz,color=clr,label=lbl)

    # Plot every graph node in cartesion space
    for i,(pos,node) in enumerate(zip(data.pos,data.x)):

        # Different colours for P-W and P-P contact
        if node[-1] == 1:
            clr = 'navy'
            lbl = 'Real particle'
        else:
            clr = 'darkred'
            lbl = 'Virtual particle'

        # Define radius for normalized x y z coordinates
        if node[0] != 0: radius = node[0]
        plotx, ploty, plotz = [pos[i]/radius for i in [0,1,2]]
        ax.scatter(plotx,ploty,plotz,color=clr,label=lbl)
    
    if manual_axes == True:
        limits = limits/radius
        #ax.set_xticks(np.arange(-1,1.1))
        ax.set_xlim(limits[0])
        #ax.set_yticks(np.arange(-1,1.1))
        ax.set_ylim(limits[1])
        #ax.set_zticks(np.arange(-2,2.1))
        ax.set_zlim(limits[2])
    ax.set(xlabel='X',ylabel="Y",zlabel="Z")
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    legend_without_duplicate_labels(ax)
    plt.show()

## Plot two particles system

# Plot coordinates in cartesian space for a given dimension
def PlotAxes(bc_rollout,real_rollout,ML_rollout,dim,ax):
    r = real_rollout[0][0,3]
    real = []
    for particles in real_rollout:
        real.append([particles[0,dim],particles[1,dim]])
    real = np.array(real)
    coorstr = ['X','Y','Z']
    ax.plot(bc_rollout[:,dim,dim]/r,'black')
    ax.plot(bc_rollout[:,dim+3,dim]/r,'black',label='Wall')
    ax.plot(real_rollout[:,0,dim]/r, 'red', label='DEM Prediction')
    ax.plot(real_rollout[:,1,dim]/r, 'blue')
    ax.plot(ML_rollout[:,0,dim]/r, 'red', linestyle='dashed', label='ML Prediction')
    ax.plot(ML_rollout[:,1,dim]/r, 'blue', linestyle='dashed')
    ax.set(xlabel='Timestep',ylabel=f'{coorstr[dim]} Coordinate (R normalized)')
    ax.set_title(f'{coorstr[dim]} Coordinate')

# Convert list of pytorch data objects to numpy array of particle data
def DataListToPositionArray(Rollout,datalist):
    real_pos = np.zeros((Rollout.timesteps,2,3))
    real_data = np.zeros((Rollout.timesteps,2,7))
    for t,data in enumerate(datalist):
        real_pos[t] = data.pos[data.mask]
        real_data[t] = data.x[data.mask]
    data_array = np.concatenate((real_pos,real_data),axis=2)
    return data_array

# Plot all three cartesion dimensions
def PlotXYZ(Rollouts,t_max):
    fig, axes = plt.subplots(1,3,sharey=True)
    fig.set_figwidth(19)
    ML_data_array = DataListToPositionArray(Rollouts,Rollouts.ML_rollout)
    DEM_data_array = DataListToPositionArray(Rollouts,Rollouts.GroundTruth)
    for i, ax in enumerate(axes):   
        PlotAxes(Rollouts.BC_rollout,
                DEM_data_array[:min(100,t_max)],
                ML_data_array[:t_max],
                i,ax)
        ax.set_xlim(xmin=0,xmax=t_max)

## Render deformed particles
def GetAllContactpoints(data):
    real_edge = data.edge_index[:,data.edge_mask]
    origins = real_edge[0,:]
    destinations = real_edge[1,:]
    midpoints = (data.pos[origins]+data.pos[destinations])/2
    wallpoints = data.pos[~data.mask]
    contactpoints = torch.concatenate((midpoints,wallpoints))
    return contactpoints

def GetContactPerParticle(data,contactpoints):
    Numpar = data.x[data.mask].shape[0]
    ParContactPoints = [[] for i in range(Numpar)]
    ParContactNormals = [[] for i in range(Numpar)]
    for i,par_i in enumerate(data.edge_index[1,:]):
        par_i = int(par_i)
        ParContactPoints[par_i].append(contactpoints[i])
        normal_temp = contactpoints[i]-data.pos[par_i]
        ParContactNormals[par_i].append(normal_temp/np.linalg.norm(normal_temp))
    ParContactPoints = [np.array(ParContactPoints[i]) for i in range(len(ParContactPoints))]    
    return ParContactPoints, ParContactNormals

def DeformedParticleMesh(radius,center,contactpoints,contactnormals,resolution=100):
    sphere = pv.Sphere(radius,center,theta_resolution=resolution,phi_resolution=resolution)
    for contactpoint,contactnormal in zip(contactpoints,contactnormals):

        contactpoint=np.expand_dims(contactpoint,0)-center
        contactnormal=np.expand_dims(contactnormal,0)

        projection = np.inner(np.array(sphere.points-center)-contactpoint,contactnormal)
        transformedpoints = np.where(projection>0,sphere.points-projection*contactnormal,sphere.points)
        sphere.points = transformedpoints
    return sphere

def RenderParticles(data):
    geom = []
    contactpoints = GetAllContactpoints(data)
    ParContactPoints, ParContactNormals = GetContactPerParticle(data,contactpoints)
    for i in range(len(ParContactPoints)):
        radius = data.x[data.mask][i][0].item()
        center = np.array(data.pos[data.mask][i])
        sphere = DeformedParticleMesh(radius,center,ParContactPoints[i],ParContactNormals[i])
        geom.append(sphere)
        geom = pv.merge(geom)
    return geom