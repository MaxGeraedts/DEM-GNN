from HeteroML import HeteroDEMDataset, TrainHetero,ForwardTrainHetero, MakeDIRs
from torch.utils.data import random_split

dataset_name    = '2Sphere'
model_ident     = 'Emb16'
retrain         = True
overfit         = False
force_reload    = True

batch_size      = 64
lr              = 0.01
epochs          = 100

push_forward_loops = 0
push_forward_epochs = 2000
push_forward_step_max_list:list = [15]*push_forward_loops

msg_num = 3
emb_dim = 128
num_layers = 3

MakeDIRs(dataset_name)
if overfit == True:
    dataset_train = HeteroDEMDataset(dataset_name,dataset_type='train',force_reload=force_reload,overfit_sim_idx=0)
    dataset_val = None
    #dataset_train, dataset_val = random_split(dataset,[0.85,0.15])
else:
    dataset = HeteroDEMDataset(dataset_name,dataset_type='train',force_reload=force_reload)
    #dataset_val = HeteroDEMDataset(dataset_name,dataset_type='validate',force_reload=force_reload)
    dataset_train,dataset_val = random_split(dataset,[0.85,0.15])

if epochs>0:
    train = TrainHetero(dataset_name,model_ident,batch_size,lr,epochs,msg_num,emb_dim,num_layers)
    train(dataset_train, dataset_val,retrain)

if push_forward_loops>0:
    train_forward = ForwardTrainHetero(dataset_name,model_ident,dataset_train,batch_size,lr,push_forward_epochs,bundle_size=1)

for push_idx, push_forward_step_max in enumerate(push_forward_step_max_list):
    train_forward(push_idx,push_forward_step_max,validate_eq=True)

