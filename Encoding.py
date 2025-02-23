import numpy as np
import os
from tqdm import tqdm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d

import torch
from torch_geometric.data import Data

# Aggregate set of simulations on the drive to a single numpy array
def AggregateRawData(data_dir,folder):
    folder_dir = os.path.join(data_dir,folder,"Results")                        # Directory for all simulation results
    top = []
    coor = []
    properties = []
    data = []
    data_start = []
    bc = []
    for name in tqdm(os.listdir(folder_dir)):                                   # For every simulation  
        # Aggregate Global properties for every simulation
        sim_dir = os.path.join(folder_dir,name)
        prop = np.loadtxt(os.path.join(sim_dir,"data_properties.dat"),ndmin=2)
        mat = np.loadtxt(os.path.join(sim_dir,"data_material.dat"))
        r = np.loadtxt(os.path.join(sim_dir,"data_radii.dat"))
        mat = mat-1
        prop_temp=np.zeros((len(mat),3))
        for i,p in enumerate(mat):
            prop_temp[i,1:3] = prop[int(p), 0:2]
            prop_temp[:,0] = r  
        properties.append(prop_temp)                                            # Shape of data         for simulation [Nstep[R E v]]]

        # Aggregate coordinates for every step
        step = 0
        coor_sim = []                                                           # Shape of Coordinates  for simulation [Nstep[Npar,[x y z]]]
        data_temp = []                                                          # Shape of Coordinates  for simulation [Nstep[Npar,[x y z]]]
        for par_dir in os.listdir(sim_dir):                                     # For every load step
            if "Particles" in par_dir:
                coor_step = np.loadtxt(os.path.join(sim_dir,par_dir)).reshape(-1,3)
                coor_sim.append(coor_step)
                data_temp.append(np.concatenate((coor_step,prop_temp),axis=1))
                step+=1
            
        coor.append(coor_sim)                                                   # Add simulation data to list of simulation data arrays
        data.append(data_temp)

        # Aggregate Start positions
        coor_start = np.loadtxt(f"{sim_dir}\\data_particle.dat")                # Shape of Coordinates for simulation [Npar,[x y z]]]
        data_start.append(np.concatenate((coor_start,prop_temp),axis=1))        # Shape of data_start for simulation [Npar,[x y z R E v]]

        # Aggregate graph topology for every step
        top_list = []                                                           
        for par_dir in os.listdir(sim_dir):                                     # For every load step
            if "PairContact" in par_dir:
                top_ar = np.loadtxt(os.path.join(sim_dir,par_dir))
                top_ar -= 1                                                     #MATLAB to Python index
                top_list.append(top_ar[:,:2].astype(int))
        top.append(top_list)      
        
        # Aggregate Boundary conditions
        bc.append(np.genfromtxt(os.path.join(sim_dir,"BC.csv"),delimiter=","))  # Boundary conditions [simulation[WallID,[x y z Nx Ny Nz dx dy dz]]]
    return data_start,data,top,bc

# Generate and encode virtual particles at BC intersections
def BCEncoding(par_step,top_step,bc_step):
    top_pw = top_step[top_step[:,1]<0,:]
    wid = (top_pw[:,1]+2)*-1                                                    # WallID converted to python index
    pid = top_pw[:,0]                                                           # Particle ID touching wall

    p = par_step[pid]                                                           # particle P which touches wall
    a = bc_step[wid,:3]-p                                                       # Vector A from particle P to point W on plane
    b = bc_step[wid,3:6]                                                        # Vector B: normal vector wall
    a1_u = np.sum((a*b),axis=1)                                                 # Vector a1(unit) : Projection A normal to wall
    a1_u = a1_u[:, np.newaxis]                                                      
    a1 = a1_u*b                                                                 # Vector a1 : Projection A normal to wall
    Pw = a1+p                                                                   # Point PW: projection P on Wall

    P_virtual = np.concatenate((Pw,
                               np.zeros((Pw.shape[0],3)),                       # Zeros for normal vector features
                               b,
                               np.zeros((Pw.shape[0],1))),                      # Ones as real particle binary classifier
                               axis=1)
    
    top_new = np.copy(top_step)
    for i,topi in enumerate(top_pw):
        top_new[-len(top_pw)+i,1] = i+len(par_step)

    return P_virtual, top_new

# Encode the Aggregated data
def Encoding(data,top,bc):
    data_enc = []
    top_enc = []
    for i, sim in enumerate(data):
        data_sim = []
        top_sim = []
        for j, step in enumerate(sim):

            # Pad real particles with zeros and add binary classifier
            P_real = np.concatenate((step,
                                     np.zeros((step.shape[0],3)),               # Zeros for normal vector features
                                     np.ones((step.shape[0],1))),               # Ones as real particle binary classifier
                                     axis=1)
            
            # Add Virtual particles to encode BC's
            bc_step = np.copy(bc[i])
            bc_step[:,:3] = bc[i][:,:3]+(j+2)*bc[i][:,-3:]                      # Update BC's
            P_virtual, top_new = BCEncoding(P_real[:,:3],top[i][j],bc_step)     # Virtual particle coordinates & Updated topology indexing
            
            data_sim.append(np.concatenate((P_real,P_virtual),axis=0))
            top_sim.append(top_new)
        data_enc.append(data_sim)  
        top_enc.append(top_sim)  
    return data_enc, top_enc

# Retrieve raw data directory on the local drive
def GetDataDir():
    user = os.path.expanduser("~")
    if user[-8:] == "20182319":
        data_dir = os.path.join(user,r"Documents\Master\Graduation\Data")
    else:
        data_dir = r"D:\Tue Files\Master\Graduation\Data"
    return data_dir

# Save the aggregated and encoded dataset
def save(dataset_name,data_enc,top_enc,data_start,bc):
    np.save(f"{os.getcwd()}\\Data\\Raw\\{dataset_name}_Data.npy",np.array(data_enc, dtype=object),allow_pickle=True)
    np.save(f"{os.getcwd()}\\Data\\Raw\\{dataset_name}_Topology.npy",np.array(top_enc, dtype=object),allow_pickle=True)
    np.save(f"{os.getcwd()}\\Data\\Raw\\{dataset_name}_Data_start",np.array(data_start, dtype=object),allow_pickle=True)
    np.save(f"{os.getcwd()}\\Data\\Raw\\{dataset_name}_BC",np.array(bc, dtype=object),allow_pickle=True)
    
# Load an aggregated and encoded dataset 
def load(dataset_name: str):
    """
    Loads aggregated data from disk
    
    Args:
        dataset_name (str): Name of the loaded dataset
    
    Returns:
        Tuple: [data_start, data, top , bc]
    """
    data = np.load(f"{os.getcwd()}\\Data\\raw\\{dataset_name}_Data.npy",allow_pickle=True)
    top = np.load(f"{os.getcwd()}\\Data\\raw\\{dataset_name}_Topology.npy",allow_pickle=True)
    data_start = np.load(f"{os.getcwd()}\\Data\\raw\\{dataset_name}_Data_start.npy",allow_pickle=True)
    bc = np.load(f"{os.getcwd()}\\Data\\raw\\{dataset_name}_BC.npy",allow_pickle=True)
    return data_start,data,top,bc

# Create array of P-W intersections
def WallParticleIntersection(point,bc,i,tol):
    topW = []
    for wid in range(len(bc)):
        a = bc[wid,:3]-point[:3]                                                    # Vector A from particle P to point W on plane
        b = bc[wid,3:6]                                                         # Vector B: normal vector wall
        a1 = np.abs(np.sum(a*b))                                                # Vector a1(unit) : Absolute size projection A normal to wall
        if a1 - point[3] <= tol*point[3]:
            topW.append([i+1,-(wid+1)])
    return topW

# Construct a topology from scratch
def ConstructTopology(par_data,bc,tol):
    topP = []
    topW = []
    top = []
    for i in range(len(par_data)):
        Xi = par_data[i,:3]
        Ri = par_data[i,3]
        for j in range(i+1,len(par_data)):
            Xj = par_data[j,:3]
            Rj = par_data[i,3]
            if np.linalg.norm(Xi-Xj)-Ri-Rj <= tol*Ri:
                topP.append([i+1,j+1])
        
        topW_temp = WallParticleIntersection(par_data[i],bc,i,tol)
        topW = np.append(topW,topW_temp)

    topP = np.array(topP).reshape((-1,2))
    topW = np.array(topW).reshape((-1,2))
    top = np.concatenate([topP,topW]).astype(int)
    return top

# From list of particles and boundaries, generate model input data
def GetEdgeIdx(top,real_idx):
    top_r=top[np.isin(top[:,1],real_idx)]
    top_v =np.flip(top,axis=1) 
    edge_index = torch.tensor(np.concatenate((top_r,top_v),axis=0),dtype=torch.long).t().contiguous()
    return edge_index 
    
def ToPytorchData(par_data,bc,tol=0):
    # Construct Topology
    top=ConstructTopology(par_data,bc,tol)
    top-=1                                                                 # Topology to python idx
    # Add virutal particles and fix topology index
    P_virtual, top_enc = BCEncoding(par_data[:,:3],top,bc)                 
    P_real = np.concatenate((par_data,
                            np.zeros((par_data.shape[0],3)),               # Zeros for normal vector features
                            np.ones((par_data.shape[0],1))),               # Ones as real particle binary classifier
                            axis=1)
    par_enc = np.concatenate((P_real,P_virtual),axis=0).astype(float)
    
    #Convert encoded data to PyTorch Data object
    real_idx = par_enc[:,-1:].squeeze().nonzero()
    maskx = np.squeeze(par_enc[:,-1:]==1)
    x = torch.tensor(par_enc,dtype=torch.float)
    edge_index = GetEdgeIdx(top_enc,real_idx) 

    data = Data(pos=x[:,:3],x=x[:,3:],edge_index=edge_index,mask=maskx)
    return data

if __name__ == "__main__":
    dataset_name = "2Sphere"
    data_start,data_agr,top_agr,bc = AggregateRawData(GetDataDir(),dataset_name)
    data_enc, top_enc = Encoding(data_agr,top_agr,bc)
    save("2Sphere",data_enc,top_enc,data_start,bc)
