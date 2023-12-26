from __future__ import absolute_import
import torch.nn as nn
import torch
from models.modulated_gcn_conv import ModulatedGraphConv
from models.graph_non_local import GraphNonLocal
from nets.non_local_embedded_gaussian import NONLocalBlock2D
from .attention import Attention
from .attention import Mlp

class _GraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None):
        super(_GraphConv, self).__init__()

        self.gconv =  ModulatedGraphConv(input_dim, output_dim, adj)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.gconv(x).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x


class _ResGraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, hid_dim, p_dropout):
        super(_ResGraphConv, self).__init__()

        self.gconv1 = _GraphConv(adj, input_dim, hid_dim, p_dropout)
        self.gconv2 = _GraphConv(adj, hid_dim, output_dim, p_dropout)

    def forward(self, x):
        residual = x
        out = self.gconv1(x)
        out = self.gconv2(out)
        return residual + out


class _GraphNonLocal(nn.Module):
    def __init__(self, hid_dim, grouped_order, restored_order, group_size):
        super(_GraphNonLocal, self).__init__()

        self.non_local = GraphNonLocal(hid_dim, sub_sample=group_size)
        self.grouped_order = grouped_order
        self.restored_order = restored_order

    def forward(self, x):
        out = x[:, self.grouped_order, :]
        out = self.non_local(out.transpose(1, 2)).transpose(1, 2)
        out = out[:, self.restored_order, :]
        return out


class ModulatedGCN(nn.Module):
    def __init__(self, adj, hid_dim, coords_dim=(2, 2), num_layers=4, nodes_group=None, p_dropout=None):
        super(ModulatedGCN, self).__init__()
        _gconv_input = [_GraphConv(adj, coords_dim[0], hid_dim, p_dropout=p_dropout)]
        _gconv_layers = []

        if nodes_group is None:
            for i in range(num_layers):
                _gconv_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
        else:
            group_size = len(nodes_group[0])
            assert group_size > 1

            grouped_order = list(reduce(lambda x, y: x + y, nodes_group))
            restored_order = [0] * len(grouped_order)
            for i in range(len(restored_order)):
                for j in range(len(grouped_order)):
                    if grouped_order[j] == i:
                        restored_order[i] = j
                        break

            _gconv_input.append(_GraphNonLocal(hid_dim, grouped_order, restored_order, group_size))
            for i in range(num_layers):
                _gconv_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
                _gconv_layers.append(_GraphNonLocal(hid_dim, grouped_order, restored_order, group_size))

        self.gconv_input = nn.Sequential(*_gconv_input)
        self.gconv_layers = nn.Sequential(*_gconv_layers)
        self.gconv_output = ModulatedGraphConv(hid_dim, coords_dim[1], adj) 
        self.non_local = NONLocalBlock2D(in_channels=hid_dim, sub_sample=False)
    def forward(self, x):
        x = x.squeeze() 
        #x = x.permute(0,2,1)
        out = self.gconv_input(x)

        out = self.gconv_layers(out)
        
        out = out.unsqueeze(2)
        out = out.permute(0,3,2,1)
        out = self.non_local(out)
        
        out = out.permute(0,3,1,2)
        out = out.squeeze()
        out = self.gconv_output(out)
        
        #out = out.permute(0,2,1)
        #out = out.unsqueeze(2)
        #out = out.unsqueeze(4)
        return out


class JT_MGCN(nn.Module):
    def  __init__(self, adj, in_feature=2, emb_dim=240, out_feature=2, hid_dim=384, p_dropout=None):
        super(JT_MGCN, self).__init__()
        self.adj=adj
        self.fc =  nn.Linear(in_feature, emb_dim)
        self.norm = nn.LayerNorm(emb_dim)
        self.gcn = ModulatedGraphConv(emb_dim, hid_dim, adj)
        self.GELU = nn.GELU()
        self.gcn2 = ModulatedGraphConv(hid_dim, emb_dim, adj)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.fc_out = nn.Linear(emb_dim, out_feature)

    def forward(self, x):
        out = self.fc(x)
        out = self.norm(out)
        out = self.gcn(out)
        out = self.GELU(out)
        out = self.gcn2(out)
        out = self.norm2(out)
        out = self.fc_out(out)
        return out


 



class MGCN(nn.Module):
    def  __init__(self, adj, emb_dim=256, MLP_dim=512, hid_dim=384, joint=17, p_dropout=None):
        super(MGCN, self).__init__()
        self.adj=adj
       
        #--------------------------------------------------------------
        self.pa=nn.Parameter(torch.zeros([1, joint, emb_dim]))
        self.L = nn.Linear(emb_dim, emb_dim)
        self.ATT = Attention(emb_dim)
        self.MLP = Mlp(emb_dim, MLP_dim, hid_dim)
        #--------------------------------------------------------------

        self.gcn = ModulatedGraphConv(emb_dim, hid_dim, adj)
        
        self.GELU = nn.GELU()
        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None


        self.gcn2 = ModulatedGraphConv(hid_dim, emb_dim, adj)
        self.norm2 = nn.LayerNorm(emb_dim)

        
    def forward(self, x):
        out = x
        out1 = self.gcn(out)
        out1 = self.GELU(out1)
        if self.dropout is not None:
            out1 = self.dropout(out1)

        out2 = self.L(out+self.pa)
        out2 = self.ATT(out2)
        out2 = self.MLP(out2)

        out = self.gcn2(out1+out2)
        out = self.norm2(out)

        return out

class JTMGCN(nn.Module):
    def  __init__(self, adj, in_feature=2, emb_dim=256, MLP_dim=512, out_feature=2, hid_dim=384, num_layers=3, joint=17, p_dropout=None):
        super(JTMGCN, self).__init__()
        
        self.fc =  nn.Linear(in_feature, emb_dim)
        self.norm = nn.LayerNorm(emb_dim)

        MGCN_layers = []
        for i in range(num_layers):
            MGCN_layers.append(MGCN(adj, emb_dim, MLP_dim, hid_dim, joint))
        
        self.MGCNs = nn.Sequential(*MGCN_layers)

        self.fc_out = nn.Linear(emb_dim, out_feature)


    def forward(self, x):
        out = self.fc(x)
        out = self.norm(out)

        out = self.MGCNs(out)
        
        out = self.fc_out(out)

        return out
