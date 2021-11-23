from Network.gated_gcn_net import GatedGCNNet

"""
    Utility file to select GraphNN model as
    selected by the user
"""



def GatedGCN(net_params):
    return GatedGCNNet(net_params)



def gnn_model(MODEL_NAME, net_params):
    models = {
        'GatedGCN': GatedGCN,

    }

    return models[MODEL_NAME](net_params)