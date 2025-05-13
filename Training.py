import torch.cuda
import torch_geometric.transforms as T

from Encoding import AggregateRawData, save
from ML_functions import DEM_Dataset, Trainer, GetModel, SaveModelInfo

print(torch.cuda.is_available())

aggregate       = False
force_reload    = False
train           = True
dataset_name    = "N400_Mono"
model_ident     = "Model_1"

if aggregate == True:
    data_dir = "/home/20182319/Data"
    ArgsAggregation = AggregateRawData(data_dir,dataset_name)
    save(dataset_name,*ArgsAggregation)

pre_transform = T.Compose([T.Cartesian(False),
                           T.Distance(norm=False,cat=True)])

dataset_train     = DEM_Dataset(dataset_name,"train"   ,'delta', force_reload, pre_transform)
dataset_val       = DEM_Dataset(dataset_name,"validate",'delta', force_reload, pre_transform)
dataset_test      = DEM_Dataset(dataset_name,"test"    ,'delta', force_reload, pre_transform)

if train == True:
    model = GetModel(dataset_name,model_ident,
                     msg_dim=64,
                     emb_dim=64,
                     edge_dim=4)
    
    SaveModelInfo(model,dataset_name,model_ident)
    
    trainer = Trainer(model, dataset_test,dataset_val,
                      batch_size=64,
                      lr=0.0000001,
                      epochs=1000,
                      model_name=f"{dataset_name}_{model_ident}")
    
    trainer.train_loop()