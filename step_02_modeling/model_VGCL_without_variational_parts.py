from torch_geometric.nn import GCNConv
from torch import nn
import torch
import torch.nn.functional as F
import os
import numpy as np


script_dir = os.path.dirname(os.path.abspath(__file__))



class Encoder_sc(nn.Module):

    def __init__(self, dims, sc_normalize=True, add_self_loops=False, use_batch_norm=True, dropout_rate=0.2):
        super().__init__()

        in_dim, hidden_dim, out_dim = dims[0], dims[1:-1], dims[-1]
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate

        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None


        self.layers.append(GCNConv(in_channels=in_dim, out_channels=hidden_dim[0], normalize=sc_normalize, add_self_loops=add_self_loops))
        if use_batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim[0]))

        for i in range(len(hidden_dim) - 1):
            self.layers.append(GCNConv(in_channels=hidden_dim[i], out_channels=hidden_dim[i+1], normalize=sc_normalize, add_self_loops=add_self_loops))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim[i+1]))

        self.outlayer = GCNConv(in_channels=hidden_dim[-1], out_channels=out_dim, normalize=sc_normalize, add_self_loops=add_self_loops)


    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_attr)
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            x = self.prelu(x)
            if self.training and self.dropout_rate > 0 and i < len(self.layers) - 1:
                x = self.dropout(x)
        return self.outlayer(x, edge_index, edge_attr)





class Encoder_fc(nn.Module):

    def __init__(self, dims, sc_normalize=True, add_self_loops=False, use_batch_norm=True, dropout_rate=0.2):
        super().__init__()

        in_dim, hidden_dim, out_dim = dims[0], dims[1:-1], dims[-1]
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate

        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None


        self.layers.append(GCNConv(in_channels=in_dim, out_channels=hidden_dim[0], normalize=sc_normalize, add_self_loops=add_self_loops))
        if use_batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim[0]))

        for i in range(len(hidden_dim) - 1):
            self.layers.append(GCNConv(in_channels=hidden_dim[i], out_channels=hidden_dim[i+1], normalize=sc_normalize, add_self_loops=add_self_loops))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim[i+1]))

        self.outlayer = GCNConv(in_channels=hidden_dim[-1], out_channels=out_dim, normalize=sc_normalize, add_self_loops=add_self_loops)


    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_attr)
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            x = self.prelu(x)
            if self.training and self.dropout_rate > 0 and i < len(self.layers) - 1:
                x = self.dropout(x)

        return self.outlayer(x, edge_index, edge_attr)







def train(model_sc, model_fc, optimizer_sc, optimizer_fc, batch_sc, batch_fc, coefs):

    batch_sc = batch_sc.to('cuda')
    batch_fc = batch_fc.to('cuda')

    model_sc.train()
    model_fc.train()

    optimizer_sc.zero_grad()
    optimizer_fc.zero_grad()

    z_sc_nodes = model_sc(batch_sc)
    z_fc_nodes = model_fc(batch_fc)

    contrastive_loss, d_loss_sc, d_loss_fc, _ = Loss(z_sc_nodes, z_fc_nodes, batch_sc, batch_fc)
    loss = coefs[0]*contrastive_loss + coefs[1]*d_loss_sc + coefs[2]*d_loss_fc
    loss.backward()



    max_grad_norm = 500
    torch.nn.utils.clip_grad_norm_(model_sc.parameters(), max_grad_norm)
    torch.nn.utils.clip_grad_norm_(model_fc.parameters(), max_grad_norm)


    # print()
    # print("\n=== SC Model Gradients ===")
    # for name, param in model_sc.named_parameters():
    #     if param.grad is not None:
    #         grad_mean = param.grad.mean().item()
    #         grad_std = param.grad.std().item()
    #         grad_norm = param.grad.norm().item()
    #         print(f"SC - {name}: mean={grad_mean:.6f}, std={grad_std:.6f}, norm={grad_norm:.6f}")
    #     else:
    #         print(f"SC - {name}: No gradient")


    # print("\n=== FC Model Gradients ===")
    # for name, param in model_fc.named_parameters():
    #     if param.grad is not None:
    #         grad_mean = param.grad.mean().item()
    #         grad_std = param.grad.std().item()
    #         grad_norm = param.grad.norm().item()
    #         print(f"FC - {name}: mean={grad_mean:.6f}, std={grad_std:.6f}, norm={grad_norm:.6f}")
    #     else:
    #         print(f"FC - {name}: No gradient")
    # print()


    optimizer_sc.step()
    optimizer_fc.step()



    return float(loss.detach()), float(contrastive_loss.detach()), float(d_loss_sc.detach()), float(d_loss_fc.detach())




def test(model_sc, model_fc, batch_sc, batch_fc, coefs):

    batch_sc = batch_sc.to('cuda')
    batch_fc = batch_fc.to('cuda')

    model_sc.eval()
    model_fc.eval()

    with torch.no_grad():
        z_sc_nodes = model_sc(batch_sc)
        z_fc_nodes = model_fc(batch_fc)

        contrastive_loss, d_loss_sc, d_loss_fc, d_nodes = Loss(z_sc_nodes, z_fc_nodes, batch_sc, batch_fc)
        loss = coefs[0]*contrastive_loss + coefs[1]*d_loss_sc + coefs[2]*d_loss_fc

        d_batch = d_nodes.reshape((-1, 360)).to('cpu')

        return float(loss.detach()), float(contrastive_loss.detach()), float(d_loss_sc.detach()), float(d_loss_fc.detach()), d_batch







def Loss(z_sc_nodes, z_fc_nodes, batch_sc, batch_fc):

    d_nodes = torch.sum((z_sc_nodes-z_fc_nodes)**2, dim=1)
    contrastive_loss = d_nodes.mean()

    d_loss_sc, d_loss_fc = Distance_Loss_sc(z_sc_nodes, batch_sc), Distance_Loss_fc(z_fc_nodes, batch_fc)

    return contrastive_loss, d_loss_sc, d_loss_fc, d_nodes




def Distance_Loss_sc(z_sc_nodes, batch_sc):

    attr = []
    for id in batch_sc.sample_id:
        id_ls = id.split('_')
        id_ls.pop(-3)
        id_ls.pop(-2)
        new_id = '_'.join(id_ls)
        attr.append(np.load(os.path.join(script_dir, 'sc_dataset', 'raw', new_id+'.npy')))
    attr = torch.tensor(np.array(attr), dtype=torch.float, device='cuda')

    d_loss_sc = torch.mean((compute_distance(z_sc_nodes, batch_sc) - (1-attr)).abs(), dim=0)
    
    return d_loss_sc.mean()


def Distance_Loss_fc(z_fc_nodes, batch_fc):

    attr = []
    for id in batch_fc.sample_id:
        attr.append(np.load(os.path.join(script_dir, 'fc_dataset', 'raw', id+'.npy')))
    attr = torch.tensor(np.array(attr), dtype=torch.float, device='cuda')

    d_loss_fc = torch.mean((compute_distance(z_fc_nodes, batch_fc) - (1-attr.abs())).abs(), dim=0)
    
    return d_loss_fc.mean()



def compute_distance(batch_z, batch):
    
    batch_size = batch.batch.max().item() + 1
    distances = torch.full((batch_size, 360, 360), -1, dtype=torch.float, device='cuda')

    for graph_idx in range(batch_size):
        graph_z = batch_z[batch.batch == graph_idx]
        distances[graph_idx] = torch.cdist(graph_z, graph_z)

    np.save('D:\\Download\\batch_distances.npy', distances.to('cpu').detach().numpy())
    
    return distances