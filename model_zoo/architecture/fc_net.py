import torch

from collections import OrderedDict

from torch.nn import Linear, BatchNorm1d, ReLU


class FCNet(torch.nn.Sequential):
    """ Basic fully-connected neural network architecture
    """

    def __init__(self, input_dim, output_dim, hidden_width, hidden_depth=4,
                 activation="relu", batch_norm=True) -> None:
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
            modules.append(("linear1", Linear(input_dim, output_dim)))
        else:
            modules.append(("linear1", Linear(input_dim, hidden_width[0])))
            for i in range(1, hidden_depth + 1):
                if batch_norm:
                    modules.append((f"bn{i}", BatchNorm1d(hidden_width[i - 1])))
                if activation == 'relu':
                    modules.append((f"relu{i}", ReLU()))
                elif activation == 'swish':
                    modules.append((f"swish{i}", Swish()))
                else:
                    raise ValueError("Unrecognized activation")
                modules.append((
                    f"linear{i + 1}",
                    Linear(
                        hidden_width[i - 1], hidden_width[i] if i != hidden_depth else output_dim
                    )
                ))
        modules = OrderedDict(modules)
        super().__init__(modules)

    def forward(self, inputs):
        assert torch.is_tensor(inputs) and inputs.dim() == 2
        return super().forward(inputs)


class Swish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return inputs * torch.sigmoid(inputs)
