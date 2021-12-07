class Own(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_dim_node = 1  # net_params['in_dim']  # node_dim (feat is an integer)
        in_dim_edge = 1  # edge_dim (feat is a float)
        n_node_in
        hidden_dim = net_params['hidden_dim']
        n_stages = net_params['n_stages']
        n_conv_per_stage = net_params['n_conv_per_stage']
        n_classes = net_params['n_classes']
        dropout = net_params['dropout']
        self.residual = net_params['residual']
        self.device = net_params['device']

        self.embedding_h = nn.Linear(in_dim_node, hidden_dim)  # node feat is an integer
        self.embedding_e = nn.Linear(in_dim_edge, hidden_dim)