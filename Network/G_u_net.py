import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from networkx import to_scipy_sparse_matrix

from Layers.unet_graph_layer import GCN, GraphUnet, Initializer, norm_g
from Loss_function.Dice_variant import *
from Layers.mlp_readout_layer import MLPReadout

class GNet(nn.Module):
    def __init__(self, net_params):
        super(GNet, self).__init__()
        self.device = net_params['device']
        self.in_dim = 1 #net_params['in_dim']
        self.l_dim = net_params['layer_dim']
        self.h_dim = net_params['hidden_dim']
        self.n_classes = net_params['n_classes']
        self.drop_n = net_params['drop_network']
        self.drop_c = net_params['drop_classifier']
        self.l_num = net_params['L']
        self.n_act = getattr(nn, net_params['activation_network'])()
        self.c_act = getattr(nn, net_params['activation_classifier'])()
        self.ks = net_params['pool_rates_layers']
        self.s_gcn = GCN(self.in_dim, self.l_dim, self.n_act, self.drop_n)
        self.g_unet = GraphUnet(
            self.ks, self.l_dim, self.l_dim, self.l_dim, self.n_act,
            self.drop_n)
        self.out_l_1 = nn.Linear(3*self.l_dim*(self.l_num+1), self.h_dim)
        self.out_l_2 = nn.Linear(self.h_dim, self.n_classes)
        self.out_drop = nn.Dropout(p=self.drop_c)
        self.MLP_layer = MLPReadout(self.l_dim, self.n_classes)
        Initializer.weights_init(self)

    def forward(self, gs, hs, es):
        a = dgl.khop_adj(gs, 1)
        hs = self.embed(a, hs)
        h_out = hs #self.classify(hs)
        return h_out

    def embed(self, gs, hs):
        o_hs = []
        #for g, h in zip(gs, hs):
        h = self.embed_one(gs, hs)
        """o_hs.append(h)
        hs = torch.stack(o_hs, 0)"""
        hs=h
        return hs

    def embed_one(self, g, h):
        g = norm_g(g)
        h = self.s_gcn(g, h)
        hs = self.g_unet(g, h)
        #h = self.readout(hs)
        h = self.MLP_layer(hs)
        return h

    def readout(self, hs):
        h_max = [torch.max(h, 0)[0] for h in hs]
        h_sum = [torch.sum(h, 0) for h in hs]
        h_mean = [torch.mean(h, 0) for h in hs]
        h = torch.cat(h_max + h_sum + h_mean)
        return h

    def classify(self, h):
        h = self.out_drop(h)
        h = self.out_l_1(h)
        h = self.c_act(h)
        h = self.out_drop(h)
        h = self.out_l_2(h)
        return F.log_softmax(h, dim=1)

    """def metric(self, logits, labels):
        loss = F.nll_loss(logits, labels)
        _, preds = torch.max(logits, 1)
        acc = torch.mean((preds == labels).float())
        return loss, acc"""
    def loss(self, pred, label):

        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label[:,0])
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes > 0).float()

        # weighted cross-entropy for unbalanced classes
        criterion = DiceLoss #nn.CrossEntropyLoss(weight=weight)
        loss = criterion.backward(criterion,pred, label)

        return loss