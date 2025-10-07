from HeteroML import HeteroDEMDataset, HeteroTrainer, HeteroDEMGNN, GetHeteroModel, TrainHetero,ForwardTrainHetero
from ML_functions import SaveTrainingInfo, SaveModelInfo
import torch

dataset_name    = 'N050_Mono'
model_ident     = 'Simfit2'
retrain         = False

batch_size      = 1
lr              = 0.1
epochs          = 50

msg_num = 5
emb_dim = 64
num_layers = 3

dataset_train, dataset_val = [HeteroDEMDataset(dataset_name,dataset_type,force_reload=False) 
                                                for dataset_type in ['train', 'validate']]
subset_train = torch.utils.data.Subset(dataset_train,[i for i in range(99)])

train = TrainHetero(dataset_name,model_ident,batch_size,lr,epochs,msg_num,emb_dim,num_layers)
train(subset_train)

train_forward = ForwardTrainHetero(dataset_name,model_ident,dataset_train,batch_size,lr,epochs,bundle_size=1)
train_forward(start_with_push=False,push_forward_step_max=15)
train_forward(start_with_push=True,push_forward_step_max=15)
train_forward(start_with_push=True,push_forward_step_max=15)
train_forward(start_with_push=True,push_forward_step_max=15)
train_forward(start_with_push=True,push_forward_step_max=15)
