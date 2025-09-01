import numpy as np
import matplotlib.pyplot as plt
import torch
import pyvista as pv
import os 
from tqdm import tqdm, trange
from typing import Type

from Evaluation import GetVolumeAndExtremeDims, GetContactForce

def PlotBoundaryBox(BC,ax,colour,linestyle,linewidth=1):
    maxdim = GetVolumeAndExtremeDims(BC)[1]
    mesh = np.array(np.meshgrid(maxdim[0],maxdim[1],maxdim[2])).T.reshape(-1,3)
    plotx, ploty, plotz = [mesh[:,dim] for dim in [0,1,2]]
    verteces = [[0,1],[0,2],[2,3],[3,1]]
    verteces += [[i,i+4] for i in range(4)]
    verteces += [[4,5],[6,7],[4,6],[5,7]]
    for vertex in verteces:
        ax.plot(plotx[vertex],ploty[vertex],plotz[vertex],c=colour,linewidth=linewidth,linestyle=linestyle)
## Plotting the topology as a graph

# Remove legend duplicates
def legend_without_duplicate_labels(ax,**kwargs):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique),**kwargs)


# Given encoded data plot graph
def PlotGraph(ax, data,limits=None,manual_axes=False,plot_lines=True,normalize=False):
    # For every contact in topology plot line
    if plot_lines == True:
        for line_idx in data.edge_index.T:

            # Define radius for normalized x y z coordinates
            if normalize == True:
                radius = data.x[0,0]
            else:
                radius=1

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
        if normalize == False: radius=1
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
    ax.set_aspect('equal')

def PlotGraphComparison(t,Rollout,sample_idx,tol,plot_lines=True,PlotBoundary=False):
    fig, axes = plt.subplots(1,2, subplot_kw={'projection': '3d'},figsize=(20,10))
    PlotGraph(axes[0],Rollout.GroundTruth[t], plot_lines=plot_lines)
    PlotGraph(axes[1],Rollout.ML_rollout[t], plot_lines=plot_lines)
    axes[0].set_title("Ground-truth",fontsize=20,fontname="Times New Roman")
    axes[1].set_title("Model",fontsize=20,fontname="Times New Roman")
    axes[1].legend(loc='lower right')
    legend_without_duplicate_labels(axes[1])
    fig.text(0.4,1,f"Graph Comparison",fontsize=25,fontname="Times New Roman",fontweight="bold")
    fig.suptitle(f"Sample: {sample_idx}, Time: {t}, Tolerance: {tol}",fontsize=20,fontname="Times New Roman")
    if PlotBoundary=='box':
        for ax in axes:
            PlotBoundaryBox(Rollout.BC_rollout[0],ax,"dimgrey","--",2)
            PlotBoundaryBox(Rollout.BC_rollout[t],ax,"black","-",2)
    return fig

## Plot two particles system

# Plot coordinates in cartesian space for a given dimension
def PlotAxes(bc_rollout,real_rollout,ML_rollout,dim,ax,normalize):
    r = real_rollout[0][0,3]
    if normalize == False:
        r = 1
    lw = 1
    coorstr = ['X','Y','Z']

    ax.plot(bc_rollout[:,0,dim,dim]/r,'black')
    ax.plot(bc_rollout[:,0,dim+3,dim]/r,'black',label='Wall')
    ax.plot(real_rollout[:,0,dim]/r, 'darkred', label='DEM: top particle')
    ax.plot(real_rollout[:,1,dim]/r, 'darkblue', label='DEM: bottom particle')
    ax.plot(ML_rollout[:,0,dim]/r, 'red', linestyle=(0, (3, 10)), label='ML: top particle', linewidth=lw,alpha=1)
    ax.plot(ML_rollout[:,1,dim]/r, 'blue', linestyle=(0, (3, 10)), label='ML: bottom particle', linewidth=lw,alpha=1)
    ax.set(xlabel='Timestep',ylabel=f'{coorstr[dim]} Coordinate (R normalized)')
    ax.set_title(f'{coorstr[dim]} Coordinate')


# Plot all three cartesion dimensions
def PlotXYZ(Rollout: object,t_max: int,normalize: bool,axs):
    ML_data_array = np.array([np.concatenate((data.pos[data.mask],data.x[data.mask]),axis=1) for data in Rollout.ML_rollout])
    DEM_data_array = np.array([np.concatenate((data.pos[data.mask],data.x[data.mask]),axis=1) for data in Rollout.GroundTruth])
    for i, ax in enumerate(axs):   
        PlotAxes(Rollout.BC_rollout,
                DEM_data_array,
                ML_data_array[:t_max],
                i,ax, normalize)
        ax.set_xlim(xmin=0,xmax=t_max)

## Render deformed particles
from Evaluation import GetAllContactpoints, GetContactPerParticle

def DeformedParticleMesh(radius,center,contactpoints,contactnormals,deformation=True,resolution=100):
    sphere = pv.Sphere(radius,center,theta_resolution=resolution,phi_resolution=resolution)
    if deformation == True:
        for contactpoint,contactnormal in zip(contactpoints,contactnormals):

            contactpoint=np.expand_dims(contactpoint,0)-center
            contactnormal=np.expand_dims(contactnormal,0)

            projection = np.inner(np.array(sphere.points-center)-contactpoint,contactnormal)
            transformedpoints = np.where(projection>0,sphere.points-projection*contactnormal,sphere.points)
            sphere.points = transformedpoints
    return sphere

def ParticleMesh(data,deformation):
    geom = []
    contactpoints = GetAllContactpoints(data)
    ParContactPoints, ParContactNormals = GetContactPerParticle(data,contactpoints)
    for i in range(len(ParContactPoints)):
        radius = data.x[data.mask][i][0].item()
        center = np.array(data.pos[data.mask][i])
        sphere = DeformedParticleMesh(radius,center,ParContactPoints[i],ParContactNormals[i],deformation)
        geom.append(sphere)
    return geom

## Plot Vectors
def AxesLimits(ax,BC):
    ax.set_xlim([BC[3,0], BC[0,0]])
    ax.set_ylim([BC[4,1], BC[1,1]])
    ax.set_zlim([BC[5,2], BC[2,2]])
    ax.set_aspect('equal')

def Plot3DVectors(ax,origin,direction):
    from mpl_toolkits.mplot3d import Axes3D
    X,Y,Z = [origin[:,i] for i in [0,1,2]]
    U, V, W = [direction[:,i] for i in [0,1,2]]
    ax.quiver(X, Y, Z, U, V, W)
    return ax

# Evaluation Visualization
## Get Stress Tensor

## Plot Intermediate vectors
def PlotContactVectorAndForce(data,BC):
    fig, axs = plt.subplots(1,2, subplot_kw={'projection': '3d'})

    contactforce = GetContactForce(data)
    contactpoints = GetAllContactpoints(data)
    contactvector = contactpoints - data.pos[data.edge_index[1,:]]
    Plot3DVectors(axs[0],data.pos[data.edge_index[1,:]],contactvector)
    Plot3DVectors(axs[1], contactpoints,(contactforce/torch.max(contactforce))/4)
    for ax in axs: AxesLimits(ax,BC)
    return fig,axs

def PlotMeshNormals(data):
    contactpoints = GetAllContactpoints(data)
    ParContactPoints, ParContactNormals = GetContactPerParticle(data,contactpoints)

    fig, ax = plt.subplots(1,1, subplot_kw={'projection': '3d'})
    for i, (parcontactpoint, parcontactnormal) in enumerate(zip(ParContactPoints,ParContactNormals)):
        
        plotx, ploty, plotz = [parcontactpoint.reshape((-1,3))[:,dim] for dim in [0,1,2]]
        parcontactnormal.shape
        Plot3DVectors(ax,data.pos[i,:].resize(1,3),parcontactnormal.reshape((-1,3))/5)
    fig.set_figheight(50)
    fig.set_figwidth(50)

def MakeGIF(datalist,gifname,fps=7,color='lightblue',deformation=False):
    plotter = pv.Plotter(notebook=False, off_screen=True)
    spheremesh = pv.merge(ParticleMesh(datalist[0],deformation=deformation))
    plotter.add_mesh(spheremesh, color=color, show_edges=False)
    plotter.camera_position = 'xz'
    plotter.open_gif(f"{os.getcwd()}\\Figures\\{gifname}.gif",fps=fps)

    for data in tqdm(datalist):
        spheremesh.points = pv.merge(ParticleMesh(data,deformation=deformation)).points
        plotter.write_frame()

    plotter.close()


from Evaluation import AggregateForces
def PlotFres(Fsum_GT,Fsum_ML):
    fig = plt.figure(figsize=(10,7))
    plt.plot(Fsum_GT,label="Ground Truth")
    plt.plot(Fsum_ML, label="Model")
    plt.title("Sum of resultant forces",fontsize=20,fontname="Times New Roman",fontweight='bold')
    plt.ylabel("Sum(Fres)(N)",fontname="Times New Roman",fontweight='bold')
    plt.xlabel("Increment",fontname="Times New Roman",fontweight='bold')
    plt.legend()
    return fig

def PlotFnormDistribution(ax,quantiles,Fnorm,color):
    t = np.arange(Fnorm.shape[0])
    for i,quantile in enumerate(quantiles):
        quantmin  = np.percentile(Fnorm,quantile,1)
        quantmax = np.percentile(Fnorm,(100-quantile),1)
        if quantile == 50:
            ax.plot(t,quantmin,'-',color=f"tab:{color}",label="Median")
        else:
            ax.fill_between(x=t, y1=quantmin, y2=quantmax, alpha=0.2, color=f"tab:{color}",label=f"{100-quantile}%")

def PlotForceDistributionComparison(Fnorm_GT,Fnorm_ML,quantiles,sharey=False):
    fig, ax = plt.subplots(1,2,figsize=(12, 5),sharey=sharey,sharex=True)
    plt.rcParams["font.family"] = "Times New Roman"
    fig.suptitle("Evolution of the Normalized Mean Resultant Force Distribution",
                fontname="Times New Roman",
                fontweight='bold',
                fontsize=20)
    
    PlotFnormDistribution(ax[0],quantiles,Fnorm_GT,"blue")
    ax[0].legend(title="Groundtruth",title_fontproperties={"size":10,"weight":"bold"})
    ax[0].set_ylabel("Fres (N)")
    ax[0].set_xlabel("Increment")

    PlotFnormDistribution(ax[1],quantiles,Fnorm_ML,"red")
    ax[1].legend(title="Model",title_fontproperties={"size":10,"weight":"bold"})
    ax[1].set_xlabel("Increment")
    ax[0].set_ylim([0,1])
    return fig, ax

from Evaluation import GetWallStress
from ML_functions import LearnedSimulator

def PlotStressComparison(Rollout:Type[LearnedSimulator],dims:list=[0,1,2],plot_ml:bool=True):

    fig, axs_temp = plt.subplots(1,len(dims),figsize=(len(dims)*5,5),sharey=True)
    if len(dims) < 3:
        axs = [0,0,0]
        for dim in dims: axs[dim] = axs_temp
    
    S_wall = GetWallStress(Rollout.GroundTruth,Rollout.BC_rollout)

    for dim in dims:
        axs[dim].plot(S_wall[:,dim]  , label="Groundtruth: Top wall"   , color="tab:blue", alpha=0.3)
        axs[dim].plot(S_wall[:,dim+3], label="Groundtruth: Bottom wall", color="tab:blue", linestyle=(0,(5,7)))
        axs[dim].set_xlabel("Increment",fontweight='bold')

    if plot_ml == True:
        S_wall = GetWallStress(Rollout.ML_rollout,Rollout.BC_rollout)
        for dim in dims:
            axs[dim].plot(S_wall[:,dim]  , label="Model: Top wall"   , color="tab:red", alpha=0.3)
            axs[dim].plot(S_wall[:,dim+3], label="Model: Bottom wall", color="tab:red", linestyle=(0,(5,7)))    

    axs[-1].legend()
    fig.suptitle("Stress on top and bottom walls",
                fontname="Times New Roman",
                fontweight='bold',
                fontsize=12)
    if len(dims) > 1:
        dimlabels = {0:'X',1:'Y',2:'Z'}
        [axs[dim].set_title(dimlabels[dim]) for dim in dims]
  
    axs[dim].set_ylabel("Stress (N/mm2)",fontweight='bold')
    fig.tight_layout()

    return fig, axs

def PlotTrainingLoss(dataset_name,model_ident,push=True):

    model_name = f"{dataset_name}_{model_ident}"
    training_loss = np.load(os.path.join(os.getcwd(),"Models",dataset_name,f"{model_name}_Training_Loss.npy"))
    validation_loss = np.load(os.path.join(os.getcwd(),"Models",dataset_name,f"{model_name}_Validation_Loss.npy"))

    if push is True:
        model_name = f"{dataset_name}_{model_ident}_Push"
        training_loss_push = np.load(os.path.join(os.getcwd(),"Models",dataset_name,f"{model_name}_Training_Loss.npy"))
        validation_loss_push = np.load(os.path.join(os.getcwd(),"Models",dataset_name,f"{model_name}_Validation_Loss.npy"))

        training_loss = np.concatenate([training_loss,training_loss_push],axis=0)
        validation_loss = np.concatenate([validation_loss,validation_loss_push],axis=0)

    plt.rcParams["font.family"] = "Times New Roman"
    fig, axs = plt.subplots(1,2,figsize=(12,5))

    for ax in axs:
        ax.plot(training_loss,label="Training Loss")
        ax.plot(validation_loss,label="Validation Loss")
        ax.set_xlabel('Epoch',fontweight='bold')
        ax.set_ylabel("Loss",fontweight='bold')
        ax.legend()
    
    
    #axs[0].set_ylim(ymin=0)
    axs[1].set_yscale('log')
    axs[1].set_title("Logarithmic Scale")
    axs[0].set_title("Linear Scale")
    axs[0].set_ylim(bottom=0)
    return fig, axs