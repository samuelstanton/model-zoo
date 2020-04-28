import torch

from torch import nn
from .fc_net import FCNet, Swish


class RecurrentNet(nn.Module):
    def __init__(self, input_dim, output_dim,
                 enc_hidden_dim, enc_depth,
                 rec_type, rec_hidden_dim, rec_depth,
                 dec_hidden_dim, dec_depth):
        super().__init__()
        self.rec_type = rec_type
        self.rec_depth = rec_depth
        self.rec_hidden_dim = rec_hidden_dim

        # self.encoder = MLP(input_dim, enc_hidden_dim, rec_hidden_dim, enc_depth)
        # self.decoder = MLP(rec_hidden_dim, dec_hidden_dim, output_dim, dec_depth)
        # self.apply(weight_init)  # Don't apply this to the recurrent unit.
        self.encoder = FCNet(input_dim, rec_hidden_dim, enc_hidden_dim, enc_depth, 'swish', batch_norm=False,
                             init='trunc_normal')
        self.encoder.add_module('final_activation', Swish())
        self.decoder = FCNet(rec_hidden_dim, output_dim, dec_hidden_dim, dec_depth, 'swish', batch_norm=False,
                             init='trunc_normal')

        self.hidden_state = None
        if rec_type is None:
            self.rec = DummyNet()
        elif rec_type == 'LSTM':
            self.rec = nn.LSTM(rec_hidden_dim, rec_hidden_dim, num_layers=rec_depth)
        elif rec_type == 'GRU':
            self.rec = nn.GRU(rec_hidden_dim, rec_hidden_dim, num_layers=rec_depth)
        else:
            raise RuntimeError("unrecognized recurrent module type")

    def __getstate__(self):
        d = self.__dict__
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self.rec.flatten_parameters()

    def init_hidden_state(self, inputs):
        assert inputs.dim() == 2
        n_batch = inputs.size(0)

        if self.rec_type is None:
            h = None
        elif self.rec_type == 'LSTM':
            h = torch.zeros(
                self.rec_depth, n_batch, self.rec_hidden_dim, device=inputs.device)
            c = torch.zeros_like(h)
            h = (h, c)
        elif self.rec_type == 'GRU':
            h = torch.zeros(
                self.rec_depth, n_batch, self.rec_hidden_dim, device=inputs.device)
        else:
            assert False

        return h

    def forward(self, inputs):
        if self.hidden_state is None:
            self.reset(inputs)
        embedded_inputs = self.encoder(inputs).unsqueeze(0)
        res, self.hidden_state = self.rec(embedded_inputs, self.hidden_state)
        res = self.decoder(res.squeeze(0))
        return res

    def reset(self, inputs=None):
        if inputs is None:
            self.hidden_state = None
        else:
            self.hidden_state = self.init_hidden_state(inputs)


class DummyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = Swish()

    def forward(self, x, h):
        return self.activation(x), h

    def flatten_parameters(self):
        pass
