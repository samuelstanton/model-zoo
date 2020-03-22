import math
import torch

from collections import OrderedDict

from torch.nn import Linear, BatchNorm1d, ReLU, Conv1d


def keras_dense_init(m):
    if isinstance(m, Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def trunc_normal_init(m):
    if isinstance(m, Linear):
        input_dim = m.weight.shape[-1]
        std = 1 / (2 * math.sqrt(input_dim))
        u_1 = torch.rand(m.weight.size()) * (1 - math.exp(-2)) + math.exp(-2)
        u_2 = torch.rand(m.weight.size())
        z = torch.sqrt(-2 * u_1.log()) * torch.cos(2 * math.pi * u_2)
        m.weight.data.copy_(std * z)

        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.)


def get_batch_layer(input_dim, output_dim, batch_dim):
    in_channels = input_dim * batch_dim
    out_channels = output_dim * batch_dim
    return Conv1d(in_channels, out_channels, kernel_size=1, groups=batch_dim)


class BatchFCNet(torch.nn.Sequential):
    """ Basic fully-connected neural network architecture
    """

    def __init__(self, input_dim, output_dim, batch_dim, hidden_width, hidden_depth=4,
                 activation="relu", batch_norm=True, init='default') -> None:
        """
        Args:
            input_dim (int)
            target_dim (int)
            hidden_depth (int)
            hidden_width (int or list): if list, len(hidden_width) = hidden_depth
            activation (str): "relu" or "swish"
            batch_norm (bool)
        """
        params = locals()
        del params['self']
        self.__dict__ = params

        if isinstance(hidden_width, list) and len(hidden_width) != hidden_depth:
            raise ValueError("hidden width must be an int or a list with len(depth)")
        elif isinstance(hidden_width, int) and hidden_depth > 0:
            hidden_width = [hidden_width] * hidden_depth
        modules = []
        if hidden_depth == 0:
            modules.append(("linear1", get_batch_layer(input_dim, output_dim, batch_dim)))
        else:
            modules.append(("linear1", get_batch_layer(input_dim, hidden_width[0], batch_dim)))
            for i in range(1, hidden_depth + 1):
                if batch_norm:
                    modules.append((f"bn{i}", BatchNorm1d(hidden_width[i - 1] * batch_dim)))
                if activation == 'relu':
                    modules.append((f"relu{i}", ReLU()))
                elif activation == 'swish':
                    modules.append((f"swish{i}", Swish()))
                else:
                    raise ValueError("Unrecognized activation")
                modules.append((
                    f"linear{i + 1}",
                    get_batch_layer(
                        hidden_width[i - 1],
                        hidden_width[i] if i != hidden_depth else output_dim,
                        batch_dim
                    )
                ))
        modules = OrderedDict(modules)
        super().__init__(modules)
        if init == 'keras':
            print("USING KERAS DENSE INITIALIZATION")
            self.apply(keras_dense_init)
        elif init == 'trunc_normal':
            print("USING TRUNCATED NORMAL INITIALIZATION")
            self.apply(trunc_normal_init)

    def forward(self, inputs):
        return super().forward(inputs)


class Swish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return inputs * torch.sigmoid(inputs)
