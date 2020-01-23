import math
import torch

from torch import Size
from torch.nn import Linear, BatchNorm1d, ReLU
from torch.utils.data import DataLoader

from collections import OrderedDict


class FC(torch.nn.Sequential):
    """
    Multi-Layer Perceptron
    """

    def __init__(
            self,
            input_shape: Size,
            output_shape: Size,
            hidden_width: int or list,
            depth: int = 4,
            activation: str = 'relu',
            batch_norm: bool = True,
            holdout_ratio: float = 0.2,
            max_epochs_since_update: int = 5,
            batch_size: int = 256,
    ) -> None:
        """
        :param input_shape:
        :param output_shape:
        :param hidden_width:
        """
        params = locals()
        del params['self']
        self.__dict__ = params
        if isinstance(hidden_width, list) and len(hidden_width) != depth:
            raise ValueError("hidden width must be an int or a list with len(depth)")
        elif isinstance(hidden_width, int) and depth > 0:
            hidden_width = [hidden_width] * depth
        modules = []
        input_width, output_width = *input_shape, *output_shape
        if depth == 0:
            modules.append(("linear1", Linear(input_width, output_width)))
        else:
            modules.append(("linear1", Linear(input_width, hidden_width[0])))
            for i in range(1, depth + 1):
                if batch_norm:
                    modules.append((f"bn{i}", BatchNorm1d(hidden_width[i - 1])))
                if activation == 'relu':
                    modules.append((f"relu{i}", ReLU()))
                elif activation == 'swish':
                    modules.append((f"swish{i}", Swish()))
                else:
                    raise ValueError("Unrecognized activation")
                modules.append(
                    (
                        f"linear{i + 1}",
                        Linear(
                            hidden_width[i - 1], hidden_width[i] if i != depth else output_width
                        ),
                    )
                )
        modules = OrderedDict(modules)
        super().__init__(modules)

    def fit(self, dataset, holdout_ratio=0.2, early_stopping=True, max_epochs=None):
        if early_stopping and holdout_ratio <= 0.:
            raise RuntimeError("holdout dataset required for early stopping")

        n_val = min(int(2048), int(holdout_ratio * len(dataset)))
        if n_val > 0:
            n_train = len(dataset) - n_val
            train_data, val_data = torch.utils.data.random_split(dataset, [n_train, n_val])
        else:
            train_data, val_data = dataset, None

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        snapshot = (1, 1e6, self.state_dict())
        metrics, snapshot = self._training_loop(
            train_data,
            val_data,
            optimizer,
            snapshot,
            max_epochs,
            early_stopping
        )

        self.eval()
        return metrics

    def _training_loop(
            self,
            train_dataset,
            val_dataset,
            optimizer,
            snapshot,
            max_epochs,
            early_stopping
    ):
        metrics = {
            'train_loss': [],
            'val_loss': [],
        }
        exit_training = False
        num_batches = math.ceil(len(train_dataset) / self.batch_size)
        epoch = 1
        avg_train_loss = None
        alpha = 2 / (num_batches + 1)
        dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        mse_fn = torch.nn.MSELoss()
        if val_dataset:
            val_x, val_y = val_dataset[:]

        while not exit_training:
            self.train()
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                pred = self(inputs)
                loss = mse_fn(pred, labels)
                loss.backward()
                optimizer.step()

                if avg_train_loss:
                    avg_train_loss = alpha * loss.detach() + (1 - alpha) * avg_train_loss
                else:
                    avg_train_loss = loss.detach()

            if val_dataset:
                with torch.no_grad():
                    self.eval()
                    pred = self(val_x)
                    val_loss = mse_fn(pred, val_y)
                    metrics['val_loss'].append(val_loss)
            conv_metric = val_loss if early_stopping else avg_train_loss

            snapshot, exit_training = self._save_best(epoch, conv_metric, snapshot)
            if exit_training or (max_epochs and epoch == max_epochs):
                break
            epoch += 1

        self.load_state_dict(snapshot[2])
        return metrics, snapshot

    def _save_best(self, epoch, holdout_loss, snapshot):
        exit_training = False
        last_update, best_loss, _ = snapshot
        improvement = (best_loss - holdout_loss) / abs(best_loss)
        if improvement > 0.01:
            snapshot = (epoch, holdout_loss.item(), self.state_dict())
        if epoch == snapshot[0] + self.max_epochs_since_update:
            exit_training = True
        return snapshot, exit_training

    @property
    def optim_param_groups(self):
        weight_decay = [0.000025, 0.00005, 0.000075, 0.000075, 0.0001]
        groups = []
        for m in self.modules():
            if isinstance(m, Linear):
                groups.append(
                    {
                        'params': [m.weight, m.bias],
                        'weight_decay': weight_decay.pop(0)
                    }
                )

        other_params = []
        for name, param in self.named_parameters():
            if 'linear' not in name:
                other_params.append(param)
        groups.append({'params': other_params})

        return groups


class Swish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return inputs * torch.sigmoid(inputs)
