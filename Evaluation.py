import torch
from tqdm import tqdm
import numpy as np

def EffectiveE(data,idx):
    poisson_ratio = data.x[idx,2]
    Youngs_mod = data.x[idx,1]
    return (1-torch.square(poisson_ratio))/Youngs_mod
    
def EffectiveStiffness(data):
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

from Plotting import GetAllContactpoints
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