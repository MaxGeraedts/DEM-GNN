import os

import numpy as np
import numpy.typing as npt
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import torch_geometric.transforms as T
import scipy.io

# Aggregate set of simulations on the drive to a single numpy array
class AggregateRawData():
    def __init__(self, dataset_name,data_dir):
        self.folder = dataset_name
        self.data_dir = data_dir
        self.folder_dir = os.path.join(data_dir,dataset_name,"Results")
        self.num_sim = len(os.listdir(self.folder_dir))
        self.sim_dirs = [os.path.join(self.folder_dir,sim_name) for sim_name in os.listdir(self.folder_dir)]
        self.timesteps = sum(["Particles" in dir for dir in os.listdir(self.sim_dirs[0])])
        
    def _ParticleData(self,sim_dir):
        prop = np.loadtxt(os.path.join(sim_dir,"data_properties.dat"),ndmin=2)
        mat = np.loadtxt(os.path.join(sim_dir,"data_material.dat")).astype(int)
        r = np.loadtxt(os.path.join(sim_dir,"data_radii.dat"))
        par_dirs = [os.path.join(sim_dir,dir) for dir in os.listdir(sim_dir) if "Particles" in dir]
        coordinates = np.array([np.loadtxt(par_dir).reshape(-1,3) for par_dir in par_dirs])

        par_data_sim = np.zeros((*coordinates.shape[:2],6))
        par_data_sim[:,:,:3] = coordinates
        par_data_sim[:,:,3] = r
        par_data_sim[:,:,-2:] = prop[mat-1][:,:2]  

        return par_data_sim  
    
    def _TopologyData(self, sim_dir):
        top_dirs = [os.path.join(sim_dir,dir) for dir in os.listdir(sim_dir) if "PairContact" in dir]
        topology_sim = [np.loadtxt(top_dir)[:,:2]-1 for top_dir in top_dirs]
        return topology_sim
    
    def _BoundaryConditions(self,sim_dir):
        if os.path.exists(os.path.join(sim_dir,"BC.csv")):
            bc_old_format = np.genfromtxt(os.path.join(sim_dir,"BC.csv"),delimiter=",")
            bc_sim= np.zeros((2,bc_old_format.shape[0],9))
            bc_sim[0,:,:6] = bc_old_format[:,:6]
            bc_sim[1,:,:3] = bc_old_format[:,-3:]

        if os.path.exists(os.path.join(sim_dir,"BC.mat")):
            bc_sim = scipy.io.loadmat('BC.mat')['BC']

    def ParticleData(self):
        par_data = np.array([self._ParticleData(sim_dir) for sim_dir in tqdm(self.sim_dirs)])
        return par_data
    
    def TopologyData(self):
        topology = [self._TopologyData(sim_dir) for sim_dir in tqdm(self.sim_dirs)]
        return topology
    
    def BoundaryConditions(self):
        boundary_conditions = [self._BoundaryConditions(sim_dir) for sim_dir in tqdm(self.sim_dirs)]
        return np.array(boundary_conditions)

# Generate and encode virtual particles at BC intersections
def BCEncoding(par_step,top_step,bc_step):
    """Encode boundary position given particles, topology and boundary conditions in a timestep

    Args:
        par_step (ndarray): particle properties in timestep [Npar, [x y z]]
        top_step (ndarray): Particle-Particle intersection topology of the graph
        bc_step (_type_): Time step boundary condition [WallID,[x y z Nx Ny Nz dx dy dz]]

    Returns:
        tuple: [P_virtual, top_new]
    """
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
def EncodeNodes(par_t,top_t,bc_t):
    """Encode real particles, Encode boundary conditions as virtual particles, return all encoded particles and updated graph topology

    Args:
        par_t (ndarray): Particle data for timestep
        top_t (ndarray): Topology for timestep
        bc_t (ndarray): Boundary condition for timestep

    Returns:
        tuple: EncodedParticles, UpdatedTopology
    """
    # Pad real particles with zeros and add binary classifier
    P_real = np.concatenate((par_t,
                             np.zeros((par_t.shape[0],3)),               # Zeros for normal vector features
                             np.ones((par_t.shape[0],1))),               # Ones as real particle binary classifier
                             axis=1)
    P_virtual, top_new = BCEncoding(P_real[:,:3],top_t,bc_t)              # Virtual particle coordinates & Updated topology indexing
    par_enc = np.concatenate((P_real,P_virtual),axis=0)
    return par_enc.astype(float),top_new

def Encoding(data,top,bc):
    """Encode aggregated data

    Args:
        data (ndarray, list): Aggregated particle data [sim,timestep,par,[x y z R E v]]
        top (ndarray, list): List or array of simulation topologies [sim,timestep [i,j]]
        bc (_type_): Boundary conditions [simulation[WallID,[x y z Nx Ny Nz dx dy dz]]]

    Returns:
        tuple: data_enc, top_enc
    """
    data_enc = []
    top_enc = []
    for i, sim in enumerate(data):
        data_sim = []
        top_sim = []
        for t, step in enumerate(sim):
            # Update BC's
            bc_step = np.copy(bc[i])
            bc_step[:,:3] = bc[i][:,:3]+(t+2)*bc[i][:,-3:]        
            # Add Virtual particles to encode BC's           
            par_enc,top_new = EncodeNodes(step,top[i][t],bc_step)
            
            data_sim.append(np.asarray(par_enc,dtype=float))
            top_sim.append(np.asarray(top_new,dtype=float))
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
def save(dataset_name,data_agr=None,top_agr=None,bc=None):
    """Saves aggregated encoded data

    Args:
        dataset_name (Array): Name of the dataset
        data_enc (Array): Encoded aggregated data
        top_enc (Array): Encoded aggregated topology
        data_start (Array): Timestep t=-1
        bc (Array): Aggregated boundary conditions
    """
    dir = os.path.join(os.getcwd(),"Data","raw")
    if data_agr is not None:
        np.save(os.path.join(dir,f"{dataset_name}_Data.npy"),np.array(data_agr, dtype=object),allow_pickle=True)
    if top_agr is not None:
        np.save(os.path.join(dir,f"{dataset_name}_Topology.npy"),np.array(top_agr, dtype=object),allow_pickle=True)
    if bc is not None:
        np.save(os.path.join(dir,f"{dataset_name}_BC"),np.array(bc, dtype=object),allow_pickle=True)
    
    
# Load an aggregated and encoded dataset 
def load(dataset_name: str):
    """
    Loads aggregated data from disk
    
    Args:
        dataset_name (str): Name of the loaded dataset
    
    Returns:
        Tuple: [data, top , bc]
    """
    data = np.load(f"{os.getcwd()}\\Data\\raw\\{dataset_name}_Data.npy",allow_pickle=True)
    top = np.load(f"{os.getcwd()}\\Data\\raw\\{dataset_name}_Topology.npy",allow_pickle=True)
    #data_start = np.load(f"{os.getcwd()}\\Data\\raw\\{dataset_name}_Data_start.npy",allow_pickle=True).astype(float)
    bc = np.load(f"{os.getcwd()}\\Data\\raw\\{dataset_name}_BC.npy",allow_pickle=True)
    return data, top ,bc

def ProjectPointsToCylinder(points,cyl):
    point_to_cyl_origin = points-cyl[:3]
    cyl_axis = cyl[3:6]
    cyl_radius = cyl[6]
    cyl_axis /= np.linalg.norm(cyl_axis)

    axis_projection = point_to_cyl_origin*cyl_axis*cyl_axis
    axis_projection += cyl[:3]
    axis_projection_to_points = points-axis_projection
    axis_projection_to_points_scaled = axis_projection_to_points/np.linalg.norm(axis_projection_to_points,axis=1,keepdims=True)*cyl_radius
    point_on_cyl = axis_projection+axis_projection_to_points_scaled
    return point_on_cyl

def ProjectPointsToHyperplane(points,plane):
    a = plane[:3]-points                                          # Vector A from particle P to point W on plane
    b = plane[3:6]                                                # Vector B: normal vector wall
    b /= np.linalg.norm(b)                                                       
    a1 = a*b*b
    point_on_plane = points+a1
    
    return point_on_plane

# Create array of P-W intersections
def WallParticleIntersection(particles: np.ndarray,bc: np.ndarray,tol: float):
    """Check if a particle intersects with a wall

    Args:
        point (Array): Properties of a single particle
        bc (Array): Boundary condition for the current timestep
        i (Int): Particle index
        tol (float): Topology tolerance as a multiple of particle radius

    Returns:
        list: Wall intersections
        """
    topology_wall = []
    for wall_id,wall in enumerate(bc[0]):
        if wall[-1] == 1:
            point_on_wall = ProjectPointsToCylinder(particles[:3],wall)
        if wall[-1] == 0:
            point_on_wall = ProjectPointsToHyperplane(particles[:3],wall)
        P_W_contact = np.linalg.norm(particles[:3]-point_on_wall,axis=1)-particles[3] <= tol*particles[3]
        P_idx = np.argwhere(P_W_contact)
        W_idx = np.ones((P_idx.shape[0],1))*-(wall_id+1)
        top_temp = np.concatenate([P_idx,W_idx],axis=1)
        if wall_id == 0:
            topology_wall = top_temp
        else:
            topology_wall = np.concatenate([topology_wall,top_temp])
    return topology_wall

# Construct a topology from scratch
def ConstructTopology(par_data,bc,tol):
    """Construct a graph topology from particle info and boundary conditions

    Args:
        par_data (ndarray): Particle data
        bc (ndarray): Boundary conditions
        tol (float): Topology tolerance as a multiple of particle radius

    Returns:
        ndarray: Array of edge indeces describing a graph topology (MatlabIDX)
    """
    topology_par, topology_wall = [], []
    for i in range(len(par_data)):
        Xi = par_data[i,:3]
        Ri = par_data[i,3]
        for j in range(i+1,len(par_data)):
            Xj = par_data[j,:3]
            Rj = par_data[j,3]
            if np.linalg.norm(Xi-Xj)-Ri-Rj <= tol*Ri:
                topology_par.append([i+1,j+1])
        
    topology_par = np.array(topology_par).reshape((-1,2))
    topology_wall = WallParticleIntersection(par_data,bc,tol)
    topology = np.concatenate([topology_par,topology_wall]).astype(int)
    return topology

def TopologyFromPlausibleTopology_old(super_topology,par_data,bc,tol):
    """From a large topology of PLAUSIBLE contacts, particle info and boundary conditions, return a topology corresponding to actual contact at the corresponding timestep

    Args:
        super_topology (array): array of indices corresponding to contact plausible to occur in the entire simulation
        par_data (array): array of particle data, including particle position and raddii 
        bc (array): Array of Boundary planes representing boundary conditions at a timestep

    Returns:
        array: Lists actual physical contacts
    """
    mask = np.zeros((super_topology.shape[0])).astype(bool)
    for contact,[i,j] in enumerate(super_topology):

        if j>0:
            [Xi,Xj] = [par_data[idx,:3] for idx in [i,j]]
            [Ri,Rj] = [par_data[idx,3] for idx in [i,j]]
            mask[contact] = np.linalg.norm(Xi-Xj)-(Ri+Rj) <= tol*(Ri+Rj)/2

        if j<0:
            Xi,Ri = par_data[i,:3],par_data[i,3]
            wid = -j-2
            a = bc[wid,:3]-Xi                                                # Vector A from particle P to point W on plane
            b = bc[wid,3:6]                                                  # Vector B: normal vector wall
            a1 = np.abs(np.sum(a*b))                                         # Vector a1(unit) : Absolute size projection A normal to wall
            mask[contact] = a1-Ri <= tol*Ri

    topology = super_topology[mask]
    return topology

def TopologySlice(super_topology_subset,par_data,idx):
    """Generate slices of particle data given a subset of a super topology

    Args:
        super_topology_subset (array): array of Particle-particle or Particle-Wall contacts
        idx (int): dimension to slice along

    Returns:
        [idx,pos,radius]: Particle indeces of slice, particle positions of slice, raddii of slice.
    """
    par_idx = super_topology_subset[:,idx]
    pos = par_data[par_idx,:3]
    radius = par_data[par_idx,3]
    return par_idx,pos,radius

def TopologyFromPlausibleTopology(super_topology,par_data,BC_t,tol):
    """From a large topology of PLAUSIBLE contacts, particle info and boundary conditions, return a topology corresponding to actual contact at the corresponding timestep

    Args:
        super_topology (array): array of indices corresponding to contact plausible to occur in the entire simulation
        par_data (array): array of particle data, including particle position and raddii 
        bc (array): Array of Boundary planes representing boundary conditions at a timestep

    Returns:
        array: Lists actual physical contacts
    """
    R_avg = par_data[:,3].mean()
    PP_mask = super_topology[:,1]>0
    super_topology_PP = super_topology[PP_mask]
    super_topology_PW = super_topology[~PP_mask]

    [idx_i,pos_i,radius_i] = TopologySlice(super_topology_PP,par_data,0)
    [idx_j,pos_j,radius_j] = TopologySlice(super_topology_PP,par_data,1)
    mask_PP = np.linalg.norm(np.ndarray.astype(pos_i-pos_j,float),axis=1)-(radius_i+radius_j) <= tol*R_avg
    topology_PP = super_topology_PP[mask_PP]

    [idx_i,pos_i,radius_i] = TopologySlice(super_topology_PW,par_data,0)
    idx_w = -super_topology_PW[:,1]-2
    PW_vector = BC_t[idx_w,:3]-pos_i
    wall_normal_vector = BC_t[idx_w,3:6]
    wall_projection = (PW_vector*wall_normal_vector)*wall_normal_vector
    dist_wall = np.linalg.norm(wall_projection.astype(float),axis=1)
    mask_PW = dist_wall-radius_i <= tol*R_avg
    topology_PW = super_topology_PW[mask_PW]

    topology = np.concatenate((topology_PP,topology_PW))
    return topology

# From list of particles and boundaries, generate model input data
def GetEdgeIdx(top,real_idx):
    """Get graph topology edge indeces in format required by pytorch

    Args:
        top (ndarray): Encoded graph topology as numpy array
        real_idx (list): list of indeces referring to real particles

    Returns:
        Tensor: Graph topology as torch tensor
    """
    top_r=top[np.isin(top[:,1],real_idx)]
    top_v =np.flip(top,axis=1) 
    edge_index = torch.from_numpy(np.concatenate((top_r,top_v),axis=0)).long().t().contiguous()
    return edge_index 
    
def ToPytorchData(par_data,bc,tol:float=0.0,topology:bool=None, label_data:bool=None,center:bool=False)->tuple[Data,npt.NDArray]:
    """Get pytorch data object from particle properties and boundary conditions

    Args:
        par_data (ndarray): Particle data for timestep
        bc (ndarray): Updated boundary conditions
        tol (float, optional): Topology construction tolerance. Defaults to 0.

    Returns:
        Data: Pytorch data object for model input
    """
    with torch.no_grad():
        if topology is None:
            topology = ConstructTopology(par_data,bc,tol)-1

        EncodedParticles, EncodedTopology = EncodeNodes(par_data,topology,bc)

        real_idx = EncodedParticles[:,-1:].squeeze().nonzero()
        RealParticleMask = np.squeeze(EncodedParticles[:,-1:]==1)
        TorchData = torch.from_numpy(EncodedParticles)
        TorchTopology = GetEdgeIdx(EncodedTopology,real_idx) 
        edge_mask = np.all(np.isin(TorchTopology, real_idx),axis=0)
        
        data = Data(pos=TorchData[:,:3],x=TorchData[:,3:],edge_index=TorchTopology,mask=RealParticleMask,edge_mask=edge_mask)
        if label_data is not None:
            data.y = torch.from_numpy(label_data.astype(float))
            
        if center == True:
            center = T.Center()
            data = center(data)

    return data, topology

def GetLength(listorarray):
    if type(listorarray) == list:
        length = len(listorarray)
    if type(listorarray) == np.ndarray:
        length = listorarray.shape[0]
    return length

def ConvertToDirected(data):
    Nreal = np.sum(data.edge_mask)
    dirmask = np.ones_like(data.edge_mask)
    dirmask[int(Nreal/2):int(Nreal)] = False
    data.edge_index = data.edge_index[:,dirmask]
    data.edge_mask = data.edge_mask[dirmask]
    return data

if __name__ == "__main__":
    dataset_name = "2Sphere"
    ArgsAggregation = AggregateRawData(GetDataDir(),dataset_name)
    save("2Sphere",*ArgsAggregation)
    #data_enc, top_enc = Encoding(*ArgsAggregation)
    
