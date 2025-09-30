from HeteroML import HeteroDEMDataset, HeteroTrainer, HeteroDEMGNN, GetHeteroModel
from ML_functions import SaveTrainingInfo, SaveModelInfo
import torch

dataset_name    = 'N400_Mono'
model_ident     = 'Simfit'
retrain         = False

batch_size      = 1
lr              = 0.1
epochs          = 2000

msg_num = 5
emb_dim = 64
num_layers = 3

dataset_train, dataset_val, dataset_test = [HeteroDEMDataset(dataset_name,dataset_type,force_reload=False) 
                                            for dataset_type in ['train', 'validate','test']]
subset_train = torch.utils.data.Subset(dataset_train,[i for i in range(99)])
print(len(subset_train))

model,msg = GetHeteroModel(dataset_name,model_ident,dataset_train[0].metadata(),
                           msg_num,emb_dim,num_layers,retrain)
print(msg)
SaveModelInfo(model,dataset_name,model_ident,hetero=True)

model_name=f"{dataset_name}_{model_ident}"
trainer = HeteroTrainer(model,batch_size,lr,epochs,dataset_name,model_ident)
trainer.train_loop(subset_train)
SaveTrainingInfo(dataset_train,trainer)
