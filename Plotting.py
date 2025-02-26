import numpy as np
import matplotlib.pyplot as plt

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

def PlotXYZ(bc_rollout,real_rollout,ML_rollout,dim,ax):
    r = real_rollout[0][0,3]
    real = []
    for particles in real_rollout:
        real.append([particles[0,dim],particles[1,dim]])
    real = np.array(real)
    coorstr = ['X','Y','Z']
    ax.plot(bc_rollout[:,dim,dim]/r,'black')
    ax.plot(bc_rollout[:,dim+3,dim]/r,'black',label='Wall')
    ax.plot(real[:,0]/r, 'red', label='DEM Prediction')
    ax.plot(real[:,1]/r, 'blue')
    ax.plot(ML_rollout[:,0,dim]/r, 'red', linestyle='dashed', label='ML Prediction')
    ax.plot(ML_rollout[:,1,dim]/r, 'blue', linestyle='dashed')
    ax.set(xlabel='Timestep',ylabel=f'{coorstr[dim]} Coordinate (R normalized)')
    ax.set_title(f'{coorstr[dim]} Coordinate')