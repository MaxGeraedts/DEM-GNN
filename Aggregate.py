from Encoding import AggregateRawData, save, GetDataDir
from HeteroML import HeteroDEMDataset
import os
dataset = False
aggregate = True

dataset_name = r"N400_MonoNeo"
data_dir = GetDataDir()

if aggregate == True:
    par_data, topology, boundary_conditions, = None, None, None
    Aggregate = AggregateRawData(dataset_name,data_dir)
    Aggregate.DetectShortSim(100)
    par_data = Aggregate.ParticleData()
    #topology = Aggregate.TopologyData()
    boundary_conditions = Aggregate.BoundaryConditions()

    save(dataset_name,par_data,topology,boundary_conditions)

if dataset == True:
    HeteroDEMDataset(dataset_name,force_reload=True)
