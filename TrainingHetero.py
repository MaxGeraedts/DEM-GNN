from HeteroML import HeteroDEMDataset, HeteroTrainer, HeteroDEMGNN, GetHeteroModel
from ML_functions import SaveTrainingInfo, SaveModelInfo
import torch

dataset_name    = '2Sphere'
model_ident     = 'Hetero'
batch_size      = 128
lr              = 0.01
epochs          = 100

msg_num = 3
emb_dim = 64
num_layers = 3

dataset_train, dataset_val, dataset_test = [HeteroDEMDataset(dataset_name,dataset_type,force_reload=False) 
                                            for dataset_type in ['train', 'validate','test']]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model,msg = GetHeteroModel(dataset_name,model_ident,device,dataset_train[0].metadata(),
                           msg_num,emb_dim,num_layers)
print(msg)
SaveModelInfo(model,dataset_name,model_ident,hetero=True)

model_name=f"{dataset_name}_{model_ident}"
trainer = HeteroTrainer(model,batch_size,lr,epochs,dataset_name,model_ident)
trainer.train_loop(dataset_train,dataset_val)
SaveTrainingInfo(dataset_train,trainer)
