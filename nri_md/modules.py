"""
Neural network modules for NRI-MD model.

Updated for Python 3.10+ and modern PyTorch compatibility.
"""

from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import my_softmax, get_offdiag_indices, gumbel_softmax

_EPS = 1e-10


class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in: int, n_hid: int, n_out: int, do_prob: float = 0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)


class CNN(nn.Module):
    """Convolutional neural network for edge classification."""

    def __init__(self, n_in: int, n_hid: int, n_out: int, do_prob: float = 0.):
        super(CNN, self).__init__()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=None, padding=0,
                                 dilation=1, return_indices=False,
                                 ceil_mode=False)

        self.conv1 = nn.Conv1d(n_in, n_hid, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(n_hid)
        self.conv2 = nn.Conv1d(n_hid, n_hid, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm1d(n_hid)
        self.conv_predict = nn.Conv1d(n_hid, n_out, kernel_size=1)
        self.conv_attention = nn.Conv1d(n_hid, 1, kernel_size=1)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Input shape: [num_sims * num_edges, num_dims, num_timesteps]
        x = F.relu(self.conv1(inputs))
        x = self.bn1(x)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        pred = self.conv_predict(x)
        attention = my_softmax(self.conv_attention(x), axis=2)

        edge_prob = (pred * attention).mean(dim=2)
        return edge_prob


class MLPEncoder(nn.Module):
    """MLP encoder for edge inference."""

    def __init__(self, n_in: int, n_hid: int, n_out: int, do_prob: float = 0., factor: bool = True):
        super(MLPEncoder, self).__init__()

        self.factor = factor

        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        if self.factor:
            self.mlp4 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
            print("Using factor graph MLP encoder.")
        else:
            self.mlp4 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
            print("Using MLP encoder.")
        self.fc_out = nn.Linear(n_hid, n_out)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, x: torch.Tensor, rel_rec: torch.Tensor, rel_send: torch.Tensor) -> torch.Tensor:
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x: torch.Tensor, rel_rec: torch.Tensor, rel_send: torch.Tensor) -> torch.Tensor:
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)
        return edges

    def forward(self, inputs: torch.Tensor, rel_rec: torch.Tensor, rel_send: torch.Tensor) -> torch.Tensor:
        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]

        x = self.mlp1(x)  # 2-layer ELU net per node

        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp2(x)
        x_skip = x

        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)
        else:
            x = self.mlp3(x)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)

        return self.fc_out(x)


class CNNEncoder(nn.Module):
    """CNN encoder for edge inference."""

    def __init__(self, n_in: int, n_hid: int, n_out: int, do_prob: float = 0., factor: bool = True):
        super(CNNEncoder, self).__init__()
        self.dropout_prob = do_prob

        self.factor = factor

        self.cnn = CNN(n_in * 2, n_hid, n_hid, do_prob)
        self.mlp1 = MLP(n_hid, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
        if self.factor:
            print("Using factor graph CNN encoder.")
        else:
            print("Using CNN encoder.")
        self.fc_out = nn.Linear(n_hid, n_out)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, x: torch.Tensor, rel_rec: torch.Tensor, rel_send: torch.Tensor) -> torch.Tensor:
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x: torch.Tensor, rel_rec: torch.Tensor, rel_send: torch.Tensor) -> torch.Tensor:
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([receivers, senders], dim=1)
        return edges

    def forward(self, inputs: torch.Tensor, rel_rec: torch.Tensor, rel_send: torch.Tensor) -> torch.Tensor:
        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]
        # New shape: [num_sims, num_atoms, num_dims, num_timesteps]
        inputs = inputs.transpose(2, 3)
        x = self.node2edge(inputs, rel_rec, rel_send)
        x = self.cnn(x)
        x = x.view(inputs.size(0), (inputs.size(1) - 1) * inputs.size(1), -1)
        # to [num_sims, num_edges, num_hidden]
        x = self.mlp1(x)
        x_skip = x

        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            x = self.mlp2(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp3(x)
        else:
            x = self.mlp2(x)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp3(x)

        return self.fc_out(x)


class MLPDecoder(nn.Module):
    """MLP decoder for trajectory prediction."""

    def __init__(self, n_in_node: int, edge_types: int, msg_hid: int, msg_out: int, 
                 n_hid: int, do_prob: float = 0., skip_first: bool = False):
        super(MLPDecoder, self).__init__()
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * n_in_node, msg_hid) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(msg_hid, msg_out) for _ in range(edge_types)])
        self.msg_out_shape = msg_out
        self.skip_first_edge_type = skip_first

        self.out_fc1 = nn.Linear(n_in_node + msg_out, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

        print('Using learned interaction net decoder.')

        self.dropout_prob = do_prob

    def single_step_forward(self, single_timestep_inputs: torch.Tensor, rel_rec: torch.Tensor, 
                          rel_send: torch.Tensor, rel_type: torch.Tensor) -> torch.Tensor:
        # single_timestep_inputs has shape
        # [batch_size, num_timesteps, num_atoms, num_dims]

        # rel_type has shape:
        # [batch_size, num_timesteps, num_atoms*(num_atoms-1), num_edge_types]

        receivers = torch.matmul(rel_rec, single_timestep_inputs)
        senders = torch.matmul(rel_send, single_timestep_inputs)
        pre_msg = torch.cat([senders, receivers], dim=-1)

        all_msgs = torch.zeros(pre_msg.size(0), pre_msg.size(1),
                              pre_msg.size(2), self.msg_out_shape,
                              device=pre_msg.device)

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0

        # Run separate MLP for every edge type
        # NOTE: To exclude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc1)):
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob, training=self.training)
            msg = F.relu(self.msg_fc2[i](msg))
            msg = msg * rel_type[:, :, :, i:i + 1]
            all_msgs += msg

        # Aggregate all msgs to receiver
        agg_msgs = torch.matmul(rel_rec.t(), all_msgs)

        # Skip connection
        aug_inputs = torch.cat([single_timestep_inputs, agg_msgs], dim=-1)

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(aug_inputs)), p=self.dropout_prob, training=self.training)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob, training=self.training)
        pred = self.out_fc3(pred)

        # Residual connection
        pred = single_timestep_inputs + pred

        return pred

    def forward(self, inputs: torch.Tensor, rel_type: torch.Tensor, 
               rel_rec: torch.Tensor, rel_send: torch.Tensor, 
               pred_steps: int = 1) -> torch.Tensor:
        # NOTE: Assumes rel_type is constant (i.e. same across all time steps).

        inputs = inputs.transpose(1, 2).contiguous()
        sizes = [rel_type.size(0), inputs.size(1), rel_type.size(1),
                rel_type.size(2)]
        rel_type = rel_type.unsqueeze(1).expand(sizes)

        time_steps = inputs.size(1)
        assert (pred_steps <= time_steps)
        preds = []

        # Only take n-th timesteps as starting points (n: pred_steps)
        last_pred = inputs[:, 0::pred_steps, :, :]
        curr_rel_type = rel_type[:, 0::pred_steps, :, :]
        # NOTE: Assumes rel_type is constant (i.e. same across all time steps).

        # Run n prediction steps
        for step in range(0, pred_steps):
            last_pred = self.single_step_forward(last_pred, rel_rec, rel_send,
                                               curr_rel_type)
            preds.append(last_pred)

        sizes = [preds[0].size(0), preds[0].size(1) * pred_steps,
                preds[0].size(2), preds[0].size(3)]

        output = torch.zeros(sizes, device=preds[0].device)
        # Re-assemble correct timeline
        for i in range(len(preds)):
            output[:, i::pred_steps, :, :] = preds[i]

        pred_all = output[:, :(inputs.size(1) - 1), :, :]

        return pred_all.transpose(1, 2).contiguous()


class RNNDecoder(nn.Module):
    """RNN decoder for trajectory prediction."""

    def __init__(self, n_in_node: int, edge_types: int, n_hid: int, 
                 do_prob: float = 0., skip_first: bool = False):
        super(RNNDecoder, self).__init__()
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * n_in_node, n_hid) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(n_hid, n_hid) for _ in range(edge_types)])
        self.msg_out_shape = n_hid
        self.skip_first_edge_type = skip_first

        self.hidden_r = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_i = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_h = nn.Linear(n_hid, n_hid, bias=False)

        self.input_r = nn.Linear(n_in_node, n_hid, bias=True)
        self.input_i = nn.Linear(n_in_node, n_hid, bias=True)
        self.input_n = nn.Linear(n_in_node, n_hid, bias=True)

        self.out_fc1 = nn.Linear(n_hid, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

        print('Using learned recurrent interaction net decoder.')

        self.dropout_prob = do_prob

    def single_step_forward(self, inputs: torch.Tensor, rel_rec: torch.Tensor, 
                          rel_send: torch.Tensor, rel_type: torch.Tensor, 
                          hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Inputs: [batch, num_atoms, num_dims]
        # Hidden: [batch, num_atoms, rnn_hidden]
        # rel_type: [batch, num_atoms*(num_atoms-1), num_edge_types]

        receivers = torch.matmul(rel_rec, inputs)
        senders = torch.matmul(rel_send, inputs)
        pre_msg = torch.cat([senders, receivers], dim=-1)

        all_msgs = torch.zeros(pre_msg.size(0), pre_msg.size(1),
                              self.msg_out_shape, device=pre_msg.device)

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0

        # Run separate MLP for every edge type
        for i in range(start_idx, len(self.msg_fc1)):
            msg = F.tanh(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob, training=self.training)
            msg = F.tanh(self.msg_fc2[i](msg))
            msg = msg * rel_type[:, :, i:i + 1]
            all_msgs += msg

        # Aggregate all msgs to receiver
        agg_msgs = torch.matmul(rel_rec.t(), all_msgs)

        # GRU-style gated aggregation
        r = torch.sigmoid(self.input_r(inputs) + self.hidden_r(agg_msgs))
        i = torch.sigmoid(self.input_i(inputs) + self.hidden_i(agg_msgs))
        n = torch.tanh(self.input_n(inputs) + r * self.hidden_h(agg_msgs))
        hidden = (1 - i) * n + i * hidden

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(hidden)), p=self.dropout_prob, training=self.training)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob, training=self.training)
        pred = self.out_fc3(pred)

        # Residual connection
        pred = inputs + pred

        return pred, hidden

    def forward(self, inputs: torch.Tensor, rel_type: torch.Tensor, 
               rel_rec: torch.Tensor, rel_send: torch.Tensor, 
               pred_steps: int = 1) -> torch.Tensor:
        inputs = inputs.transpose(1, 2).contiguous()

        time_steps = inputs.size(1)
        hidden = torch.zeros(inputs.size(0), inputs.size(2), self.msg_out_shape,
                           device=inputs.device)

        pred_all = []

        for step in range(0, inputs.size(1) - 1):
            if step == 0:
                ins = inputs[:, step, :, :]
            else:
                ins = pred_all[step - 1]

            pred, hidden = self.single_step_forward(ins, rel_rec, rel_send,
                                                   rel_type, hidden)
            pred_all.append(pred)

        preds = torch.stack(pred_all, dim=1)

        return preds.transpose(1, 2).contiguous()


class SimulationDecoder(nn.Module):
    """Simulation decoder that just returns the input (for testing)."""

    def __init__(self, loc_max: float, loc_min: float, vel_max: float, vel_min: float, suffix: str):
        super(SimulationDecoder, self).__init__()
        self.loc_max = loc_max
        self.loc_min = loc_min
        self.vel_max = vel_max
        self.vel_min = vel_min

        print('Using simulation decoder.')

    def forward(self, inputs: torch.Tensor, rel_type: torch.Tensor, 
               rel_rec: torch.Tensor, rel_send: torch.Tensor, 
               pred_steps: int = 1) -> torch.Tensor:
        return inputs[:, :, 1:, :]