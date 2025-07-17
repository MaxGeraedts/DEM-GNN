import torch
import torch_geometric.transforms as T
from tqdm import tqdm, trange
import numpy as np
from typing import Literal
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
    relative_displacement = GetGamma(data)
    Eff_Stiffness = EffectiveStiffness(data)
    Fij_size = Eff_Stiffness*(relative_displacement*torch.sqrt(relative_displacement))
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

def GetWallForce(data):
    F_wall = np.zeros((6,3))
    Fcontact = GetContactForce(data).numpy()
    F_PWcontact = Fcontact[np.invert(data.edge_mask),:]

    Wallcontact = data.MatlabTopology[data.MatlabTopology[:,1]<0]
    Wallcontact[:,1] += 2
    Wallcontact[:,1] *= -1
    Wall_idx = Wallcontact[:,1]

    for i,Wall in enumerate(Wall_idx):
        F_wall[Wall,:] += F_PWcontact[i,:]

    return F_wall

def GetWallStress(datalist,BC_rollout):
    A_wall = np.array([np.squeeze(GetWallArea(BC)) for BC in BC_rollout])
    F_wall = np.array([np.linalg.norm(GetWallForce(data),axis=1) for data in datalist])
    S_wall = F_wall/A_wall
    return S_wall

def NumpyGroupby(group_key,group_value):
    index_sort = group_key.argsort()
    key_sort = group_key[index_sort]
    value_sort = group_value[index_sort,:]

    index_split = np.unique(key_sort,return_index=True)[1][1:]
    value_grouped = np.split(value_sort,index_split)

    return value_grouped

def NormalizedResultantForce(data):
    force = GetContactForce(data).numpy()
    contact = data.edge_index.T.numpy()

    force_grouped = NumpyGroupby(group_key=contact[:,1],group_value=force)

    Fres_vectors = np.array([np.sum(group,axis=0) for group in force_grouped])
    Fres_norm = np.linalg.norm(Fres_vectors,axis=1)

    F_vectors_norm = [np.linalg.norm(group,axis=1,keepdims=True) for group in force_grouped]
    F_vectors_norm_sum = [np.sum(group).item() for group in F_vectors_norm]

    Fres_size_normalized  = Fres_norm/F_vectors_norm_sum
    return Fres_size_normalized

def AggregatedRollouts(dataset_name:str,model_ident:str,AggregatedArgs:tuple):
    model = GetModel(f"{dataset_name}_{model_ident}")[0]
    scale_name = f"{dataset_name}_bund{model.bundle_size}"
    transform = T.Compose([T.Cartesian(False),T.Distance(norm=False,cat=True),NormalizeData(scale_name)])
    Simulation = LearnedSimulator(model, scale_function = Rescale(scale_name),transform = transform)

    datalist_ML = []
    datalist_GT = []
    for sample_idx in trange(AggregatedArgs[0].shape[0]):
        Simulation.Rollout(*AggregatedArgs,sample_idx)
        datalist_ML.append(Simulation.ML_rollout)
        datalist_GT.append(Simulation.GroundTruth)
    return datalist_ML, datalist_GT

def DatalistToArray(datalist):
    pos_array = np.array([[data.pos[data.mask] for data in simulation] for simulation in datalist])
    property_array = np.array([[data.x[data.mask,:3] for data in simulation] for simulation in datalist])
    return pos_array, property_array

def GeometricMetrics(pos_test,pos_ML,radii):
    [pos_test,pos_ML,radii] = [torch.from_numpy(arg) for arg in [pos_test,pos_ML,radii]]

    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(pos_ML,pos_test)

    dist = (pos_ML-pos_test).pow(2).sum(dim=-1).sqrt()
    dist_mean = dist.mean()

    normalized_dist = dist/radii
    normalized_dist_mean = normalized_dist.mean()
    return loss.item(), dist_mean.item(), normalized_dist_mean.item()

class Evaluation:
    def __init__(self,mode:Literal["geometric","mechanics_sum","mechanics_mean"],print_results:bool=False,show_tqdm = False):
        self.mode = mode
        self.print_results = print_results
        self.show_tqdm = show_tqdm
        self._disable_tqdm = not show_tqdm

        if mode == "mechanics_sum":
            self.aggregation_function = np.sum
            self.description = "Mean sum of normalized resultantant forces"
        if mode == "mechanics_mean":
            self.aggregation_function = np.mean
            self.description = "Mean of normalized resultantant forces"
        else:
            self.description = "Geometric measures"

        if self.print_results == True:
            print(self.description)

    def EvaluateGeometric(self,datalist_ML,datalist_GT):
        pos_ML = DatalistToArray(datalist_ML)[0]
        pos_GT, prop_GT = DatalistToArray(datalist_GT)
        radii = prop_GT[:,:,:,0]

        loss, dist_mean, normalized_dist_mean = GeometricMetrics(pos_GT,pos_ML,radii)

        metrics = {'L2 Norm:': loss, 
                   "Mean Euclidean distance:": dist_mean, 
                   "Radius normalized mean Euclidean distance:": normalized_dist_mean}
        
        return metrics
    
    def EvaluateMechanics(self,datalist_ML,datalist_GT):
        model_prediction= np.array([[self.aggregation_function(NormalizedResultantForce(data)) for data in datalist_sample] for datalist_sample in tqdm(datalist_ML,disable=self._disable_tqdm)])
        metrics = {"Model:": np.mean(model_prediction).item()}

        if datalist_GT is not None:
            ground_truth    = np.array([[self.aggregation_function(NormalizedResultantForce(data)) for data in datalist_sample] for datalist_sample in tqdm(datalist_GT,disable=self._disable_tqdm)])
            metrics["Ground truth:"]  = np.mean(ground_truth).item()
         
        return metrics

    def __call__(self,datalist_ML,datalist_GT):
        if self.mode == "geometric":
            metrics = self.EvaluateGeometric(datalist_ML,datalist_GT)
        else:
            metrics = self.EvaluateMechanics(datalist_ML,datalist_GT)

        if self.print_results == True:
            [print(f"{metric:<50}{value:4f}") for metric, value in metrics.items()]   

        return metrics    
    
def CompareModels(dataset_name:str, model_idents:list[str], Evaluation_function):
    metric_dict = {}
    AggregatedArgs = MaskTestData(dataset_name,"test")

    for i, model_ident in enumerate(model_idents):
        datalist_ML, datalist_GT = AggregatedRollouts(dataset_name,f"{model_ident}_Push",AggregatedArgs)
        if i == 0:
            metrics = Evaluation_function(datalist_ML, datalist_GT)
        else:
            metrics = Evaluation_function(datalist_ML, None)

        for (key,value) in metrics.items():
            if key == "Model:":
                metric_dict[model_ident] = value
            else:
                metric_dict['Ground truth'] = value
    
    print("\n",Evaluation_function.description,"\n") 
    print(f"{"Ground truth":<50}{metrics["Ground truth"]:.10f}","\n")
    for metric, value in metrics.items():
        if metric != "Ground truth": print(f"{metric:<50}{value:.3f}")

    return metric_dict

def CoordinationNumber(datalist):
    contact_num = np.array([data.edge_index.shape[1] for data in datalist])
    par_num = np.array([data.mask.sum().item() for data in datalist])
    coordination_number = contact_num/par_num
    return coordination_number