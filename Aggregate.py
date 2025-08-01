from Encoding import AggregateRawData, save
import os

dataset_name = r"N400_Mono"

data_dir = r"D:\TUE\Master\Graduation\Data"
if os.path.exists(data_dir) != True:
    data_dir = os.path.join("..","Data")

par_data, topology, boundary_conditions, = None, None, None
Aggregate = AggregateRawData(dataset_name,data_dir)
par_data = Aggregate.ParticleData()
#topology = Aggregate.TopologyData()
boundary_conditions = Aggregate.BoundaryConditions()

save(dataset_name,par_data,topology,boundary_conditions)