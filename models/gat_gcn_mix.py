import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

# GCN-CNN based model

class GAT_GCN(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):

        super(GAT_GCN, self).__init__()

        self.n_output = n_output
        self.conv1 = GATConv(num_features_xd, num_features_xd, heads=10)
        self.conv11 = GATConv(num_features_xd*10, num_features_xd, heads=10)
        # self.conv11 = GATConv(num_features_xd*10, num_features_xd, heads=10)
        self.conv2 = GCNConv(num_features_xd, num_features_xd )
        self.conv21 = GCNConv(num_features_xd, num_features_xd )
        # self.conv21 = GCNConv(num_features_xd, num_features_xd )
        self.fc_g1 = torch.nn.Linear(num_features_xd *10*2,output_dim)
        self.fc_g2 = torch.nn.Linear(num_features_xd *2,output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 1D convolution on protein sequence
        self.conv_xt1 = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=1, kernel_size=32),
                                       nn.ReLU())

        self.fc1_xt = nn.Linear(594, output_dim*2)
        # combined layers
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 256)
        self.out = nn.Linear(256, self.n_output)  # n_output = 1 for regression task

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target
        # print('x shape = ', x.shape)
        x1 = self.conv1(x, edge_index)
        x1 = self.conv11(x1, edge_index)
        x1 = self.relu(x1)
        x1 = torch.cat([gmp(x1, batch), gap(x1, batch)], dim=1)
        x1 = self.relu(self.fc_g1(x1))
        x1 = self.dropout(x1)



        x2 = self.conv2(x, edge_index)
        x2 = self.conv2(x2, edge_index)
        x2 = self.relu(x2)
        x2 = torch.cat([gmp(x2, batch), gap(x2, batch)], dim=1)
        x2 = self.relu(self.fc_g2(x2))
        x2 = self.dropout(x2)
        # apply global max pooling (gmp) and global mean pooling (gap)
        x = torch.cat([x1,x2], dim=1)
        # x = self.fc_g2(x)

        conv_xt = self.conv_xt1(target)

        # flatten
        xt = conv_xt.view(-1, 594)
        xt = self.fc1_xt(xt)

        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
