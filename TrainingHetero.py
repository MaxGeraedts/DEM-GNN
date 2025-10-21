from HeteroML import HeteroDEMDataset, TrainHetero,ForwardTrainHetero, MakeDIRs
from torch.utils.data import random_split

overfit = True

dataset_name    = 'N400_MonoNeo'
model_ident     = 'Overfit_noval'
retrain         = True

batch_size      = 1
lr              = 0.01
epochs          = 4000

push_forward_loops = 0
push_forward_epochs = 2000
push_forward_step_max_list:list = [15]*push_forward_loops

msg_num = 5
emb_dim = 128
num_layers = 3

MakeDIRs(dataset_name)
if overfit == True:
    dataset_train = HeteroDEMDataset(dataset_name,dataset_type='train',force_reload=True,overfit_sim_idx=0)
    dataset_val = None
    #dataset_train, dataset_val = random_split(dataset,[0.85,0.15])
else:
    dataset_train = HeteroDEMDataset(dataset_name,dataset_type='train',force_reload=True)
    dataset_val = HeteroDEMDataset(dataset_name,dataset_type='validate',force_reload=True)

train = TrainHetero(dataset_name,model_ident,batch_size,lr,epochs,msg_num,emb_dim,num_layers)
train(dataset_train, dataset_val,retrain)

train_forward = ForwardTrainHetero(dataset_name,model_ident,dataset_train,batch_size,lr,push_forward_epochs,bundle_size=1)

for push_idx, push_forward_step_max in enumerate(push_forward_step_max_list):
    train_forward(push_idx,push_forward_step_max,validate_eq=True)

