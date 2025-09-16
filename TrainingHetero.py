from HeteroML import HeteroDEMDataset, HeteroTrainer, HeteroDEMGNN, GetHeteroModel
from ML_functions import SaveTrainingInfo, SaveModelInfo
import torch

dataset_name    = 'N400_mono'
model_ident     = 'Overfit'
retrain         = True

batch_size      = 1
lr              = 0.01
epochs          = 1000

msg_num = 5
emb_dim = 128
num_layers = 2

dataset_train, dataset_val, dataset_test = [HeteroDEMDataset(dataset_name,dataset_type,force_reload=False) 
                                            for dataset_type in ['train', 'validate','test']]
subset_train = torch.utils.data.Subset(dataset_train,[0])

model,msg = GetHeteroModel(dataset_name,model_ident,dataset_train[0].metadata(),
                           msg_num,emb_dim,num_layers,retrain)
print(msg)
SaveModelInfo(model,dataset_name,model_ident,hetero=True)

model_name=f"{dataset_name}_{model_ident}"
trainer = HeteroTrainer(model,batch_size,lr,epochs,dataset_name,model_ident)
trainer.train_loop(subset_train)
SaveTrainingInfo(dataset_train,trainer)
