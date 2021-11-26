from Network.gated_gcn_net import GatedGCNNet
from Network.my_gcn_net import GNet

"""
    Utility file to select GraphNN model as
    selected by the user
"""



def GatedGCN(net_params):
    return GatedGCNNet(net_params)

def GUNet(net_params):

    return GNet(net_params)

def gnn_model(MODEL_NAME, net_params):

    models = {
        'GatedGCN': GatedGCN,
        'GNet' : GUNet,

    }

    return models[MODEL_NAME](net_params)