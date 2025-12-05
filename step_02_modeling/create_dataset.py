import numpy as np
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import dense_to_sparse
import torch
import os
from tqdm import tqdm


class ConnDataset(Dataset):
    def __init__(self, conn_type, transform=None, pre_transform=None, pre_filter=None):

        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.conn_type = conn_type
        self.root = os.path.join(self.script_dir, f'{conn_type}_dataset')

        super().__init__(self.root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        if self.conn_type == 'fc':
            for _, _, files in os.walk(os.path.join(self.root, 'raw')):
                return files
        elif self.conn_type == 'sc':
            file_names = []
            for _, _, files in os.walk(os.path.join(self.root, 'raw')):
                for file in files:
                    for code in ['r1_LR', 'r1_RL', 'r2_LR', 'r2_RL']:
                        file_names.append(file.replace('log01.npy', f'{code}_log01.npy'))
            return file_names

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(3992)]

    def download(self):
        pass

    def process(self):

        idx = 0
        
        # x = np.load(os.path.join(self.script_dir, 'HCP_MMP1_centroids.npy'))
        x = np.eye(360)
        # x = np.random.rand(360, 3)

        x = torch.tensor(x, dtype=torch.float)

        for raw_path in tqdm(self.raw_paths, total=len(self.raw_paths)):
            
            raw_path_ls = raw_path.split('_')
            sample_id = raw_path.split("\\")[-1].split(".")[0]

            if self.conn_type == 'sc':
                raw_path_ls.pop(-3)
                raw_path_ls.pop(-2)
                raw_path = '_'.join(raw_path_ls)

            adj_matrix = np.load(raw_path)
            adj_matrix = torch.tensor(adj_matrix, dtype=torch.float)

            if self.conn_type == 'sc':

                percentile = 0.1

                mask = np.eye(adj_matrix.shape[0], dtype=bool)
                adj_matrix[mask] = 0

                threshold = torch.quantile(adj_matrix[~mask], 1-percentile)
                adj_matrix[adj_matrix < threshold] = 0

            elif self.conn_type == 'fc':

                percentile = 0.2

                # mask = np.eye(adj_matrix.shape[0], dtype=bool)
                # adj_matrix[mask] = 0

                # pos_threshold = torch.quantile(adj_matrix[adj_matrix > 0], 1-percentile)
                # neg_threshold = -torch.quantile(adj_matrix[adj_matrix < 0], 1-percentile)

                # pos_mask = (adj_matrix > pos_threshold) & (adj_matrix > 0)
                # neg_mask = (adj_matrix < neg_threshold) & (adj_matrix < 0)

                # new_adj_matrix = torch.zeros_like(adj_matrix)
                # new_adj_matrix[pos_mask] = adj_matrix[pos_mask]
                # new_adj_matrix[neg_mask] = adj_matrix[neg_mask]

                # new_adj_matrix[mask] = 1
                # adj_matrix = new_adj_matrix

                mask = np.eye(adj_matrix.shape[0], dtype=bool)

                threshold = torch.quantile(torch.abs(adj_matrix[~mask]), 1-percentile)
                adj_matrix[torch.abs(adj_matrix) < threshold] = 0

            edge_index, edge_attr = dense_to_sparse(adj_matrix)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, sample_id=sample_id)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1

    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx): 
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'), weights_only=False)
        return data
    
    def compute_shortest_path_length(self, adj_matrix):
        adj_matrix = adj_matrix.numpy()
        shortest_path = np.full((360, 360), -1)
        shortest_path_attrs = np.zeros((360, 360))
        for start_node in range(360):
            distances, attrs = [-1] * 360, [0] * 360
            distances[start_node] = 0
            queue = [start_node]
            while queue:
                current = queue.pop(0)
                for neighbor in range(360):
                    if adj_matrix[current][neighbor] != 0 and distances[neighbor] == -1:
                        distances[neighbor] = distances[current] + 1
                        attrs[neighbor] = attrs[current] + adj_matrix[current, neighbor]
                        queue.append(neighbor)
            shortest_path[start_node] = np.array(distances)
            shortest_path_attrs[start_node] = np.array(attrs)
        mask = np.eye(360, dtype=bool)
        shortest_path[mask] = 1
        return shortest_path_attrs / shortest_path
