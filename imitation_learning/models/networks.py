import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, BatchNorm, GCNConv
from torch_geometric.utils import to_networkx
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import LayerNorm
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing
from egnn_pytorch import EGNN
from egnn_pytorch import EGNN_Network
from .egnn import EGNN
import matplotlib.pyplot as plt
import networkx as nx

from .masked_transformer_encoder import MaskedTransformerEncoder

from .mappings import Mapping


class Tokenizer(torch.nn.Module):
    def __init__(self, env_name, output_dim, output_activation=None, device="cuda"):
        super(Tokenizer, self).__init__()

        self.mapping = Mapping(env_name)
        self.map = self.mapping.get_map()

        self.output_dim = output_dim
        self.output_activation = output_activation

        self.device = device

        self.zero_token = torch.nn.Parameter(
            torch.zeros(1, output_dim), requires_grad=False
        )

        base = lambda input_dim: torch.nn.Sequential(
            torch.nn.Linear(input_dim, output_dim)
        )
        self.tokenizers = torch.nn.ModuleDict()
        for k in self.map.keys():
            self.tokenizers[k] = base(len(self.map[k][0]))
            if output_activation is not None:
                self.tokenizers[k] = torch.nn.Sequential(
                    self.tokenizers[k], output_activation
                )

    def forward(self, x):
        x = self.mapping.create_observation(x)
        outputs = []
        for key in x.keys():
            inputs = x[key].to(self.device)
            if inputs.shape[-1] == 0:
                outputs.append(
                    self.zero_token.expand(*inputs.shape[:-1], -1).unsqueeze(1)
                )
            else:
                outputs.append(self.tokenizers[key](inputs).unsqueeze(1))
        return torch.cat(outputs, dim=1)


class Detokenizer(torch.nn.Module):
    def __init__(
        self,
        env_name,
        embedding_dim,
        action_dim,
        num_layers=1,
        global_input=False,
        output_activation=None,
        device="cuda",
    ):
        super(Detokenizer, self).__init__()

        self.mapping = Mapping(env_name)
        self.map = self.mapping.get_map()

        self.nbodies = len(self.map.keys())

        self.embedding_dim = embedding_dim
        self.action_dim = action_dim
        self.output_activation = output_activation

        self.device = device

        base = lambda output_dim: torch.nn.Linear(embedding_dim, output_dim)
        self.detokenizers = torch.nn.ModuleDict()
        if global_input:
            self.detokenizers["global"] = base(action_dim)
            if output_activation is not None:
                self.detokenizers["global"] = torch.nn.Sequential(
                    self.detokenizers["global"], output_activation
                )
        else:
            for k in self.map.keys():
                self.detokenizers[k] = base(len(self.map[k][1]))
                if output_activation is not None:
                    self.detokenizers[k] = torch.nn.Sequential(
                        self.detokenizers[k], output_activation
                    )

    def forward(self, x, weights=None):

        if "global" in self.detokenizers:
            return self.detokenizers["global"](x.to(self.device))

        action = torch.zeros(x.shape[0], self.action_dim).to(self.device)
        for i, k in enumerate(self.map.keys()):
            curr_action = self.detokenizers[k](x[:, i, :])
            action[:, self.map[k][1]] = curr_action
        return action

    def weighted_sum(self, x, weights):
        return torch.sum(x * weights.unsqueeze(-1), dim=1)  # (B, action_dim)

    def weighted_sum_per_time(self, x, weights):
        return torch.sum(x * weights.unsqueeze(-1), dim=1)  # (B, action_dim)


class Transformer(torch.nn.Module):
    def __init__(
        self,
        nbodies,
        input_dim,
        dim_feedforward=256,
        nhead=6,
        nlayers=3,
        use_positional_encoding=False,
    ):
        super(Transformer, self).__init__()
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=0.0,
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

        self.output_dim = input_dim
        self.use_positional_encoding = use_positional_encoding

        if use_positional_encoding:
            print("Using positional encoding")
            self.embed_absolute_position = nn.Embedding(
                nbodies, embedding_dim=input_dim
            )

    def forward(self, x):
        if self.use_positional_encoding:
            _, nbodies, _ = x.shape
            limb_indices = torch.arange(0, nbodies, device=x.device)
            limb_idx_embedding = self.embed_absolute_position(limb_indices)
            x = x + limb_idx_embedding
        x = self.encoder(x)
        return x


class BodyTransformer(Transformer):
    def __init__(
        self,
        nbodies,
        env_name,
        input_dim,
        dim_feedforward=256,
        nhead=6,
        num_layers=3,
        is_mixed=True,
        use_positional_encoding=False,
        first_hard_layer=1,
        random_mask=False,
    ):
        super(BodyTransformer, self).__init__(
            nbodies,
            input_dim,
            dim_feedforward,
            nhead,
            use_positional_encoding=use_positional_encoding,
        )
        shortest_path_matrix = Mapping(env_name).shortest_path_matrix
        adjacency_matrix = shortest_path_matrix < 2

        if random_mask:
            num_nonzero = torch.sum(adjacency_matrix) - adjacency_matrix.shape[0]
            prob_nonzero = num_nonzero / (
                adjacency_matrix.shape[0] * adjacency_matrix.shape[0]
            )
            adjacency_matrix = torch.rand(adjacency_matrix.shape) > prob_nonzero
            adjacency_matrix.fill_diagonal_(True)

        self.nbodies = adjacency_matrix.shape[0]

        # We assume (B x nbodies x input_dim) batches
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=0.0,
        )

        self.encoder = MaskedTransformerEncoder(encoder_layer, num_layers=num_layers)

        self.is_mixed = is_mixed
        self.use_positional_encoding = use_positional_encoding

        self.first_hard_layer = first_hard_layer

        self.register_buffer("adjacency_matrix", adjacency_matrix)

    def forward(self, x):
        if self.use_positional_encoding:
            limb_indices = torch.arange(0, self.nbodies, device=x.device)
            limb_idx_embedding = self.embed_absolute_position(limb_indices)
            x = x + limb_idx_embedding

        x = self.encoder(
            x,
            mask=~self.adjacency_matrix,
            is_mixed=self.is_mixed,
            return_intermediate=False,
            first_hard_layer=self.first_hard_layer,
        )

        return x


class BodyNet(torch.nn.Module):
    def __init__(
        self,
        env_name,
        net,
        action_dim,
        embedding_dim,
        output_activation=None,
        global_input=False,
        fixed_std=0.1,
        device="cuda",
    ):
        super(BodyNet, self).__init__()

        self.std = fixed_std
        # Pass device to Graph_Creator to keep tensors on the same device
        self.graph_creator = Graph_Creator(env_name, device=device)

        self.tokenizer = Tokenizer(
            env_name, embedding_dim, output_activation=output_activation, device=device
        )
        self.net = net
        self.detokenizer = Detokenizer(
            env_name,
            net.output_dim,
            action_dim,
            device=device,
            global_input=global_input,
        )

        self.tokenizer.to(device)
        self.net.to(device)
        # Ensure detokenizer is on the same device
        self.detokenizer.to(device)

    def forward(self, x):
        weights = None

        x = self.tokenizer(x)  # (B, nbodies, embedding_dim)
        if isinstance(self.net, GNNModule):
            x = self.graph_creator(x)
        x = self.net(x)

        x = self.detokenizer(x, weights)

        return x

    def mode(self, x):
        return self.forward(x)

    def log_prob(self, x, action):

        mu = self.forward(x)
        std = self.std
        return torch.distributions.Normal(mu, std).log_prob(action).sum(1)


class Graph_Creator(nn.Module):
    """
    Creates a PyTorch Geometric graph from observation data.

    Args:
        env_name (str): Name of the environment.
        device (str, optional): Device to use for computations ('cuda' or 'cpu'). Defaults to 'cuda'.
    """

    def __init__(self, env_name, device="cuda", k_hop: int = 1, use_virtual_node: bool = True):
        super().__init__()
        self.device = device
        self.use_virtual_node = use_virtual_node
        self.k_hop = k_hop
        self.mapping = Mapping(env_name)
        sp = self.mapping.shortest_path_matrix
        adjacency_matrix = sp <= k_hop
        # Remove diagonal to avoid duplicating self-loops (GAT will add them as needed)
        adjacency_matrix = adjacency_matrix.clone()
        adjacency_matrix.fill_diagonal_(False)
        # Base directed edges (since adjacency is symmetric, both directions are present)
        self.edge_index = adjacency_matrix.nonzero().t().contiguous().to(device)
        # Simple edge attributes (scalar 1.0); placeholder for richer features if needed
        self.edge_attr = torch.ones((self.edge_index.size(1), 1), device=device)

    def forward(self, obs: torch.Tensor, visualize=False) -> Batch:
        """
        Generates a graph batch from observation data.

        Args:
            obs (torch.Tensor): Observation tensor.
            visualize (bool, optional): Whether to visualize the graph. Defaults to False.

            Returns:
            Batch: A batch of PyTorch Geometric graphs.
        """
        batch_size = obs.shape[0]

        # Build and process graphs
        graph = self._build_batch_graphs(
            obs,
            self.edge_index,
            self.edge_attr,
            batch_size,
        )

        if visualize:
            self._visualize_graph(graph)

        return graph

    def _build_batch_graphs(
        self,
        node_features,
        edge_index,
        edge_attr,
        batch_size,
    ):
        graphs_list = []
        for b in range(batch_size):
            x_b = node_features[b]
            N = x_b.size(0)
            if self.use_virtual_node:
                # Use mean-pooled node as virtual node feature (no extra params)
                v_feat = x_b.mean(dim=0, keepdim=True)
                x_b_ext = torch.cat([x_b, v_feat], dim=0)
                v_idx = torch.tensor([N], device=x_b.device, dtype=edge_index.dtype)
                nodes = torch.arange(N, device=x_b.device, dtype=edge_index.dtype)
                # Virtual node connections (both directions)
                v_to_nodes = torch.stack([v_idx.expand_as(nodes), nodes])
                nodes_to_v = torch.stack([nodes, v_idx.expand_as(nodes)])
                eindex = torch.cat([edge_index.to(x_b.device), v_to_nodes, nodes_to_v], dim=1)
                eattr = torch.cat([
                    edge_attr.to(x_b.device),
                    torch.ones((2 * N, 1), device=x_b.device)
                ], dim=0)
            else:
                x_b_ext = x_b
                eindex = edge_index.to(x_b.device)
                eattr = edge_attr.to(x_b.device)
            graphs_list.append(Data(x=x_b_ext, edge_index=eindex, edge_attr=eattr))
        graphs = Batch.from_data_list(graphs_list)
        return graphs

    def _visualize_graph(self, graph):
        """
        Visualizes the graph in 3D.

        Args:
            graph (Batch): A batch of PyTorch Geometric graphs.
        """
        pos = graph.pos.cpu().detach().numpy()
        edge_index = graph.edge_index.cpu().detach().numpy()

        # Remove self-loops for cleaner visualization
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for i, key in enumerate(self.mapping.get_map().keys()):
            ax.scatter(pos[i, 0], pos[i, 1], pos[i, 2], label=key)
        for i in range(edge_index.shape[1]):
            ax.plot(
                [pos[edge_index[0, i], 0], pos[edge_index[1, i], 0]],
                [pos[edge_index[0, i], 1], pos[edge_index[1, i], 1]],
                [pos[edge_index[0, i], 2], pos[edge_index[1, i], 2]],
                color="black",
            )
        # Add xyz coordinate axes
        ax.quiver(0, 0, 0, 1, 0, 0, color="r")
        ax.quiver(0, 0, 0, 0, 1, 0, color="g")
        ax.quiver(0, 0, 0, 0, 0, 1, color="b")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.legend()  # Add a legend to identify nodes
        plt.show()


class GNNModule(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers=3,
        heads=4,
        dropout=0.2,
        use_virtual_node: bool = True,
        dropedge_p: float = 0.1,
        residual: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = input_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.use_virtual_node = use_virtual_node
        self.dropedge_p = dropedge_p
        self.residual = residual

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        # Input layer
        self.conv_layers.append(
            GATv2Conv(input_dim, hidden_dim, heads=heads, dropout=dropout, edge_dim=1)
        )
        self.norm_layers.append(LayerNorm(hidden_dim * heads))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.conv_layers.append(
                GATv2Conv(
                    hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout, edge_dim=1
                )
            )
            self.norm_layers.append(LayerNorm(hidden_dim * heads))

        # Output layer
        self.conv_layers.append(
            GATv2Conv(hidden_dim * heads, input_dim, heads=1, dropout=dropout, edge_dim=1)
        )
        self.norm_layers.append(LayerNorm(input_dim))

        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, data):
        from torch_geometric.utils import dropout_edge
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, 'edge_attr', None)

        if self.dropedge_p and self.dropedge_p > 0.0:
            edge_index, edge_mask = dropout_edge(edge_index, p=self.dropedge_p, training=self.training)
            if edge_attr is not None:
                edge_attr = edge_attr[edge_mask]

        for i in range(self.num_layers):
            out = self.conv_layers[i](x, edge_index, edge_attr)
            out = self.norm_layers[i](out)
            if i < self.num_layers - 1:
                out = F.gelu(out)
            # Residual connection only when feature dimensions match
            if self.residual and out.shape == x.shape:
                x = x + out
            else:
                x = out

        x_dense, mask = to_dense_batch(x, batch)
        if self.use_virtual_node:
            x_dense = x_dense[:, :-1, :]  # drop virtual node before returning

        return x_dense
