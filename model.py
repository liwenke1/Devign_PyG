import torch
from torch_geometric.nn import GatedGraphConv
from torch import nn
import torch.nn.functional as f


class DevignModel(nn.Module):
    def __init__(self, input_dim, output_dim, max_edge_types, num_steps):
        super(DevignModel, self).__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.max_edge_types = max_edge_types
        self.num_timesteps = num_steps
        self.ggnn = GatedGraphConv(out_channels=output_dim, num_layers=num_steps)
        self.conv_l1 = torch.nn.Conv1d(output_dim, output_dim, 3)
        self.maxpool1 = torch.nn.MaxPool1d(3, stride=2)
        self.conv_l2 = torch.nn.Conv1d(output_dim, output_dim, 1)
        self.maxpool2 = torch.nn.MaxPool1d(2, stride=2)

        self.concat_dim = input_dim + output_dim
        self.conv_l1_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 3)
        self.maxpool1_for_concat = torch.nn.MaxPool1d(3, stride=2)
        self.conv_l2_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 1)
        self.maxpool2_for_concat = torch.nn.MaxPool1d(2, stride=2)

        self.batchnorm_1d = torch.nn.BatchNorm1d(output_dim)
        self.batchnorm_1d_for_concat = torch.nn.BatchNorm1d(self.concat_dim)
        
        self.mlp_z = nn.Linear(in_features=self.concat_dim, out_features=1)
        self.mlp_y = nn.Linear(in_features=output_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def de_batchify_graphs(self, features=None):
        vectors = [torch.tensor(1)]
        vectors[0] = features.clone().requires_grad_(True)
        output_vectors = torch.stack(vectors)

        return output_vectors

    def forward(self, input):
        x, edge_index, edge_types = input.x.cuda(), input.edge_index.cuda(), input.edge_attr.cuda()
        output = self.ggnn(x, edge_index, edge_types)
        x_i = self.de_batchify_graphs(x)
        h_i = self.de_batchify_graphs(output)
        c_i = torch.cat((h_i, x_i), dim=-1)
        Y_1 = self.maxpool1(
            f.relu(
                self.batchnorm_1d(
                    self.conv_l1(h_i.transpose(1, 2))
                )
                
            )
        )
        Y_2 = self.maxpool2(
            f.relu(
                self.batchnorm_1d(
                    self.conv_l2(Y_1)
                )
                
            )
        ).transpose(1, 2)
        Z_1 = self.maxpool1_for_concat(
            f.relu(
                self.batchnorm_1d_for_concat(
                    self.conv_l1_for_concat(c_i.transpose(1, 2))
                )
            )
        )
        Z_2 = self.maxpool2_for_concat(
            f.relu(
                self.batchnorm_1d_for_concat(   
                    self.conv_l2_for_concat(Z_1)
                )
            )
        ).transpose(1, 2)
        before_avg = torch.mul(self.mlp_y(Y_2), self.mlp_z(Z_2))
        avg = before_avg.mean(dim=1)
        result = self.sigmoid(avg).squeeze(dim=-1)
        return result