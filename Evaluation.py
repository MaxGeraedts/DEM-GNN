import torch
from tqdm import tqdm, trange
import numpy as np
from ML_functions import LearnedSimulator, NormalizeData, GetModel, Rescale, NormalizePos, MaskTestData

def GetAllContactpoints(data:object):
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
    ParContactNormals = [np.array(ParContactNormals[i]) for i in range(len(ParContactNormals))]  
    return ParContactPoints, ParContactNormals

def EffectiveE(data,idx):
    poisson_ratio = data.x[idx,2]
    Youngs_mod = data.x[idx,1]
    return (1-torch.square(poisson_ratio))/Youngs_mod
    
def EffectiveStiffness(data:object):
    """Calculate relative stiffnes Nij for all contacts

    Args:
        data (object): Graph describing particle assembly

    Returns:
        array: 1xNcontact
    """
    i,j = data.edge_index[1,:], data.edge_index[0,:]
    E_ij = EffectiveE(data,i)
    E_ij[data.edge_mask]+=EffectiveE(data,j[data.edge_mask])

    R_ij = 1/data.x[i,0] 
    R_ij[data.edge_mask]+=1/data.x[j[data.edge_mask],0]

    Nij = (4/3)*1/(E_ij)*(1/torch.sqrt(R_ij))
    return Nij

def GetGamma(data):
    """Calculate relative displacement for all contacts

    Args:
        data (object): Graph describing particle assembly

    Returns:
        array: 3xNcontact
    """
    i,j = data.edge_index[1,:], data.edge_index[0,:]
    R_i = data.x[i,0]
    R_i += data.x[j,0]
    gamma = R_i - torch.norm(data.pos[i]-data.pos[j],dim=1)
    gamma[gamma<0] = 0
    return gamma

def GetContactForce(data):
    """Calculate contact force vectors for all contacts

    Args:
        data (object): Graph describing particle assembly

    Returns:
        array: 3xNcontact array
    """
    i,j = data.edge_index[1,:], data.edge_index[0,:]
    gamma = GetGamma(data)
    Nij = EffectiveStiffness(data)
    Fij_size = Nij*(gamma*torch.sqrt(gamma))
    contactnormal = (data.pos[i]-data.pos[j])
    contactnormal = torch.div(contactnormal.T,torch.norm(contactnormal,dim=1))
    Fij = Fij_size*contactnormal
    return Fij.T

def GetVolumeAndExtremeDims(BC):
    """Given a boundary condition array at time t, calculate volume

    Args:
        BC (Array): Boundary condition hyperplane points at timestep

    Returns:
        float: Boundary volume
    """
    xmax, xmin = BC.item(0,0), BC.item(3,0)
    ymax, ymin = BC.item(1,1), BC.item(4,1)
    zmax, zmin = BC.item(2,2), BC.item(5,2)
    maxdim = np.array([[xmin,xmax],[ymin,ymax],[zmin,zmax]])
    vol = (xmax-xmin)*(ymax-ymin)*(zmax-zmin)
    return vol,maxdim

def GetStressTensor(data,BC):
    """Calculate Internal stress tensor for one timestep

    Args:
        data (Data object): Graph describing particle assembly at time t
        BC (array): Boundary condition at time t

    Returns:
        array: 3x3 Gauchy stress tensor
    """
    contactpoints = GetAllContactpoints(data)
    contactvector = contactpoints - data.pos[data.edge_index[1,:]]
    #contactvector = data.pos[data.edge_index[1,:]] - data.pos[data.edge_index[0,:]]

    contactforce = GetContactForce(data)
    stress_tensor = torch.zeros((3,3))
    vol = GetVolumeAndExtremeDims(BC)[0]
    for contact in range(contactforce.shape[0]):
        Fij = contactforce[contact].reshape(3,1)
        Lij = contactvector[contact].reshape(1,3)
        stress_tensor += Fij*Lij
    stress_tensor /= vol
    return stress_tensor

from Encoding import ConvertToDirected
def GetInternalStressRollout(Rollout):
    """Calculate internal stress tensor (Gauchy) for every timestep

    Args:
        Rollout (list): List of Torch Data objects describing particle assemblies

    Returns:
        array: [Timestep[3x3]]
    """
    stress_evo = torch.zeros((Rollout.timesteps,3,3))
    for t in range(Rollout.timesteps):
        data = Rollout.GroundTruth[t]
        data = ConvertToDirected(data.clone())
        BC = Rollout.BC_rollout[t][:,:3]
        stress_evo[t] = GetStressTensor(data,BC)
    return stress_evo

def AggregateForces(datalist):
    """Calculates resultant forces for each particles, their norms and the sum of all norms

    Args:
        datalist (list): Rollout list of data describing graphs

    Returns:
        Farg (tuple): F_agr: resultant force for each particle, F_norm: norms of resultant forces, F_sum: sum of all norms
    """
    F_res = torch.zeros([datalist.shape[0],datalist[0].mask.sum(),3])
    F_contact = [None]*datalist.shape[0]
    for t,data in tqdm(enumerate(datalist)):
        F_contact[t] = GetContactForce(data)
        for i,par_index in enumerate(data.edge_index[1,:]):
            F_res[t,par_index,:] += F_contact[t][i,:]
    F_norm = torch.norm(F_res,dim=2)
    F_sum = torch.sum(F_norm,1)
    return F_contact,F_res, F_norm, F_sum

def GetWallArea(BC):
    maxdims = GetVolumeAndExtremeDims(BC)[1]
    [x,y,z] = maxdims[:,1] - maxdims[:,0]
    x,y,z = x.item(),y.item(),z.item()
    A_wall = [[y*z,x*z,x*y,y*z,x*z,x*y]]
    return np.array(A_wall)

def WallReaction(datalist,BC_rollout,Fcontact):
    F_wall = torch.zeros((BC_rollout[0].shape[0],len(datalist),3))
    S_wall = torch.zeros_like(F_wall)
    for t,data in enumerate(datalist): 
        F_PWcontact = Fcontact[t][np.invert(data.edge_mask),:]
        Wallcontact = data.MatlabTopology[data.MatlabTopology[:,1]<0]
        Wallcontact[:,1] += 2
        Wallcontact[:,1] *= -1
        Wall_idx = Wallcontact[:,1]
        F_PWcontact.shape, Wall_idx.shape
        A_wall = GetWallArea(BC_rollout[t])
        for i,Wall in enumerate(Wall_idx):
            F_wall[Wall,t,:] += F_PWcontact[i,:]
        S_wall[:,t,:] = F_wall[:,t,:]/A_wall.T
    return F_wall, S_wall

def AggregatedRollouts(AggregatedArgs,Simulation):
    pos_test = torch.from_numpy(np.array(AggregatedArgs[0][:,:,:,:3],float))
    pos_ML = torch.zeros_like(pos_test)

    for sample_idx in trange(pos_test.shape[0]):
        Simulation.Rollout(*AggregatedArgs,sample_idx)
        pos_ML[sample_idx,:,:] = torch.stack([data.pos[data.mask] for data in Simulation.ML_rollout]) 

    return pos_test, pos_ML

def EvaluateAggregatedRollouts(pos_test,pos_ML,par_data):
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(pos_ML,pos_test)

    dist = (pos_ML-pos_test).pow(2).sum(dim=-1).sqrt()
    dist_mean = dist.mean()

    radii = torch.from_numpy(np.array(par_data[:,:,:,3],float))
    normalized_dist = dist/radii
    normalized_dist_mean = normalized_dist.mean()
    return loss, dist_mean, normalized_dist_mean

def Evaluate(dataset_name,model_ident,transform,AggregatedArgs):
    model = GetModel(f"{dataset_name}_{model_ident}")[0]
    Simulation = LearnedSimulator(model, scale_function = Rescale(dataset_name),transform = transform)

    pos_test, pos_ML = AggregatedRollouts(AggregatedArgs,Simulation)
    loss, dist_mean, normalized_dist_mean = EvaluateAggregatedRollouts(pos_test,pos_ML,AggregatedArgs[0])

    contents = {'L2 Norm:': loss, "Mean Euclidean distance:": dist_mean, "Radius normalized mean Euclidean distance:": normalized_dist_mean}
    for metric, value in contents.items():
        print(f"{metric:<50}{value:4f}")
    return loss, dist_mean, normalized_dist_mean