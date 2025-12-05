from torch import nn
from torch_geometric.nn import GCNConv
import torch


class Encoder_sc(nn.Module):

    def __init__(self, dims, mlp_dims, sc_normalize=True, add_self_loops=False, use_batch_norm=True, dropout_rate=0.2):
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

        self.mlp = nn.Sequential(nn.Linear(mlp_dims[0], mlp_dims[1]),
                                 nn.ReLU(),
                                 nn.Linear(mlp_dims[1], mlp_dims[2]))


    def forward(self, data, label_edge_index):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_attr)
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            x = self.prelu(x)
            if self.training and self.dropout_rate > 0 and i < len(self.layers) - 1:
                x = self.dropout(x)
        x = self.outlayer(x, edge_index, edge_attr)
        x1 = x[label_edge_index[0]]
        x2 = x[label_edge_index[1]]
        x = torch.cat((x1, x2), dim=-1)
        x = self.mlp(x)
        return x




def train(model_sc, optimizer_sc, batch_sc, batch_fc):

    batch_sc = batch_sc.to('cuda')
    batch_fc = batch_fc.to('cuda')

    model_sc.train()

    optimizer_sc.zero_grad()

    pFC = model_sc(batch_sc, batch_fc.edge_index).view(-1)

    loss = nn.MSELoss()(pFC, batch_fc.edge_attr) + 0.0001*(torch.norm(model_sc.mlp[0].weight, 2) + torch.norm(model_sc.mlp[2].weight, 2))
    loss.backward()

    # max_grad_norm = 500
    # torch.nn.utils.clip_grad_norm_(model_sc.parameters(), max_grad_norm)


    print()
    print("\n=== SC Model Gradients ===")
    for name, param in model_sc.named_parameters():
        if param.grad is not None:
            grad_mean = param.grad.mean().item()
            grad_std = param.grad.std().item()
            grad_norm = param.grad.norm().item()
            print(f"SC - {name}: mean={grad_mean:.6f}, std={grad_std:.6f}, norm={grad_norm:.6f}")
        else:
            print(f"SC - {name}: No gradient")


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



    return float(loss.detach())




def test(model_sc, batch_sc, batch_fc, returnR=False):

    batch_sc = batch_sc.to('cuda')
    batch_fc = batch_fc.to('cuda')

    model_sc.eval()

    with torch.no_grad():
        pFC = model_sc(batch_sc, batch_fc.edge_index).view(-1)

        loss = nn.MSELoss()(pFC, batch_fc.edge_attr) + 0.0001*(torch.norm(model_sc.mlp[0].weight, 2) + torch.norm(model_sc.mlp[2].weight, 2))

        if returnR:

            pFC_batch = torch.zeros((batch_sc.x.shape[0], 360), device='cuda', dtype=torch.float32)
            eFC_batch = torch.zeros((batch_sc.x.shape[0], 360), device='cuda', dtype=torch.float32)
            edge_index = batch_fc.edge_index
            for i in range(pFC_batch.shape[0]):
                pFC_batch[i, edge_index[1, edge_index[0]==i]%360] = pFC[edge_index[0]==i]
                eFC_batch[i, edge_index[1, edge_index[0]==i]%360] = batch_fc.edge_attr[edge_index[0]==i]

            r_batch = row_pearson_correlation(pFC_batch, eFC_batch).reshape(-1, 360).to('cpu')

            return float(loss.detach()), r_batch
        
        else:

            return float(loss.detach())
    


def row_pearson_correlation(x, y):

    x_centered = x - x.mean(dim=1, keepdim=True)
    y_centered = y - y.mean(dim=1, keepdim=True)
    
    covariance = (x_centered * y_centered).sum(dim=1)
    x_std = x_centered.norm(dim=1)
    y_std = y_centered.norm(dim=1)

    corr = covariance / (x_std * y_std + 1e-8)

    return corr