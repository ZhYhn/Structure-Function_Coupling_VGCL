from torch_geometric.nn import VGAE
from create_dataset import ConnDataset
from torch_geometric.loader import DataLoader
import torch
from torch.utils.data import Subset
from tqdm import tqdm
import math
from model_v3_VGAE_differentMetric import Encoder_sc, Encoder_fc, train, test
from torch import nn
import numpy as np
import random




def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=0.25, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)







def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything()







# ====== params for GCNConv ======
sc_normalize = True
fc_normalize = True
add_self_loops = False

# ====== params for BatchNorm and Dropout ======
use_batch_norm = True
dropout_rate = 0.2

# ====== params for Encoder ======
sc_dims = [360, 128, 32]
fc_dims = [360, 128, 32]

# ====== coefs of loss ======
coefs = [1, 10, 1, 15, 1] # contrastive_loss, d_loss_sc, kl_loss_sc, d_loss_fc, kl_loss_fc

# ====== params for Adam ======
sc_lr, fc_lr = 0.01, 0.01
sc_weight_decay, fc_weight_decay = 1e-3, 1e-3

# ====== params for Training ======
total_size = 3992 // 4

num_folds = 10
test_size = int(num_folds/100 * total_size)

batch_size = 16
num_epochs = 2






subjs_test_all = []
d_test_all = torch.tensor([])
log = []


for i in range(num_folds):


    model_sc = VGAE(Encoder_sc(sc_dims, sc_normalize, add_self_loops, use_batch_norm, dropout_rate)).to('cuda')
    model_fc = VGAE(Encoder_fc(fc_dims, fc_normalize, add_self_loops, use_batch_norm, dropout_rate)).to('cuda')


    model_sc.apply(init_weights)
    model_fc.apply(init_weights)



    total_indices = torch.arange(total_size)

    if i == num_folds-1 :

        train_indices = total_indices[:i*test_size]
        test_indices = total_indices[i*test_size:]

    else:

        train_indices_l = total_indices[:i*test_size]
        test_indices = total_indices[i*test_size:(i+1)*test_size]
        train_indices_r = total_indices[(i+1)*test_size:]

        train_indices = torch.concat([train_indices_l, train_indices_r])



    new_train_indices = torch.concat([4*train_indices, 4*train_indices+1, 4*train_indices+2, 4*train_indices+3])
    new_test_indices = torch.concat([4*test_indices, 4*test_indices+1, 4*test_indices+2, 4*test_indices+3])

    train_indices = new_train_indices[torch.randperm(new_train_indices.size(0))]
    test_indices = new_test_indices[torch.randperm(new_test_indices.size(0))]



    dataset_sc = ConnDataset(conn_type='sc')
    dataset_fc = ConnDataset(conn_type='fc')

    sc_train_dataset = Subset(dataset_sc, train_indices)
    sc_test_dataset = Subset(dataset_sc, test_indices)

    fc_train_dataset = Subset(dataset_fc, train_indices)
    fc_test_dataset = Subset(dataset_fc, test_indices)



    indices = torch.randperm(len(sc_train_dataset))

    sc_train_dataloader = DataLoader(sc_train_dataset, batch_size=batch_size, sampler=indices)
    sc_test_dataloader = DataLoader(sc_test_dataset, batch_size=batch_size, shuffle=False)

    fc_train_dataloader = DataLoader(fc_train_dataset, batch_size=batch_size, sampler=indices)
    fc_test_dataloader = DataLoader(fc_test_dataset, batch_size=batch_size, shuffle=False)

    optimizer_sc = torch.optim.Adam(model_sc.parameters(), lr=sc_lr, weight_decay=sc_weight_decay)
    optimizer_fc = torch.optim.Adam(model_fc.parameters(), lr=fc_lr, weight_decay=fc_weight_decay)








    for epoch in range(num_epochs):
        
        
        print()

        head = f'Fold {i+1}/{num_folds}  Epoch {epoch+1}/{num_epochs}:'
        print(head)

        log.append(head)
        log.append("")

        print()


        train_nums = len(sc_train_dataset)
        for idx, (batch_sc, batch_fc) in enumerate(zip(sc_train_dataloader, fc_train_dataloader)):

            assert [sc_id[:6] for sc_id in batch_sc.sample_id] == [fc_id[:6] for fc_id in batch_fc.sample_id]
            
            loss, contrastive_loss, d_loss_sc, kl_loss_sc, d_loss_fc, kl_loss_fc = train(model_sc, model_fc, optimizer_sc, optimizer_fc, batch_sc, batch_fc, coefs)
            print_train_loss = f'{idx+1}/{math.ceil(train_nums/batch_size)}  loss: {loss:.4f}, contrastive loss: {contrastive_loss:.4f}, d loss sc: {d_loss_sc:.4f}, kl loss sc: {kl_loss_sc:.4f}, d loss fc: {d_loss_fc:.4f}, kl loss fc: {kl_loss_fc:.4f}'
            print(print_train_loss)
            print()

            log.append(print_train_loss)
            log.append("")

        indices = torch.randperm(len(sc_train_dataset))
        sc_train_dataloader = DataLoader(sc_train_dataset, batch_size=batch_size, sampler=indices)
        fc_train_dataloader = DataLoader(fc_train_dataset, batch_size=batch_size, sampler=indices)





        test_nums = len(sc_test_dataset)
        num_batches = math.ceil(test_nums/batch_size)
        total_loss, total_contrastive_loss, total_d_loss_sc, total_kl_loss_sc, total_d_loss_fc, total_kl_loss_fc = 0, 0, 0, 0, 0, 0
        d_test = torch.tensor([])
        subjs_test = []
        for batch_sc, batch_fc in tqdm(zip(sc_test_dataloader, fc_test_dataloader), total=num_batches):

            assert [sc_id[:6] for sc_id in batch_sc.sample_id] == [fc_id[:6] for fc_id in batch_fc.sample_id]

            loss, contrastive_loss, d_loss_sc, kl_loss_sc, d_loss_fc, kl_loss_fc, d_batch = test(model_sc, model_fc, batch_sc, batch_fc, coefs)

            total_loss += loss
            total_contrastive_loss += contrastive_loss
            total_d_loss_sc += d_loss_sc
            total_kl_loss_sc += kl_loss_sc
            total_d_loss_fc += d_loss_fc
            total_kl_loss_fc += kl_loss_fc

            d_test = torch.concat([d_test, d_batch], dim=0)
            subjs_test += batch_sc.sample_id


        print_test_loss = f'Loss: {total_loss/num_batches:.4f}, Contrastive Loss: {total_contrastive_loss/num_batches:.4f}, D Loss SC: {total_d_loss_sc/num_batches:.4f}, KL Loss SC: {total_kl_loss_sc/num_batches:.4f}, D Loss FC: {total_d_loss_fc/num_batches:.4f}, KL Loss FC: {total_kl_loss_fc/num_batches:.4f}'
        
        print(print_test_loss)
        
        print()

        log.append(print_test_loss)
        log.append("")
        log.append("")

    
    log.append("")

    subjs_test_all += subjs_test
    d_test_all = torch.concat([d_test_all, d_test.mean(dim=0, keepdim=True)], dim=0)



with open('D:\\Download\\cross_validation_log.txt', 'w') as f:
    for row in log:
        f.write(row + "\n")


np.save('D:\\Download\\distances.npy', d_test_all.to('cpu').detach().numpy())

with open('D:\\Download\\subjs.txt', 'w') as f:
    for subj in subjs_test_all:
        f.write(subj + '\n')