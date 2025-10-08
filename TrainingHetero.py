from HeteroML import HeteroDEMDataset, HeteroTrainer, HeteroDEMGNN, GetHeteroModel, TrainHetero,ForwardTrainHetero
from torch.utils.data import random_split

dataset_name    = 'N400_Mono'
model_ident     = 'b4'
retrain         = True

batch_size      = 4
lr              = 0.01
epochs          = 2000

push_forward_loops = 4
push_forward_epochs = 2000
push_forward_step_max_list:list = [15]*push_forward_loops

msg_num = 5
emb_dim = 128
num_layers = 3

dataset = [HeteroDEMDataset(dataset_name,dataset_type,force_reload=True,overfit_sim_idx=0) 
                            for dataset_type in ['train']]

dataset_train, dataset_val = random_split(dataset,[0.85,0.15])

train = TrainHetero(dataset_name,model_ident,batch_size,lr,epochs,msg_num,emb_dim,num_layers)
train(dataset_train, dataset_val,retrain)

train_forward = ForwardTrainHetero(dataset_name,model_ident,dataset,batch_size,lr,push_forward_epochs,bundle_size=1)

for push_idx, push_forward_step_max in enumerate(push_forward_step_max_list):
    train_forward(push_idx,push_forward_step_max,validate_eq=True)

