import math
import torch

from copy import deepcopy

from torch.nn import Linear, Parameter
from torch.nn.functional import softplus
from torch.utils.data import DataLoader
from torch.distributions import Normal
from torch.optim import Adam

from model_zoo.architecture import FCNet


class FCRegression(FCNet):
    """ Fully-connected neural network regression model w/ Gaussian predictive distributions

    Behavior should closely mimic that of the component networks in Kurtland Chua's
    bootstrapped deep ensemble implementation (https://tinyurl.com/vl4alu9)
    """

    def __init__(self, input_dim, target_dim, hidden_width, hidden_depth=4,
                 activation="relu", batch_norm=True, max_epochs_since_update=5) -> None:
        """
        Args:
            input_dim (int)
            target_dim (int)
            hidden_depth (int)
            hidden_width (int or list): if list, len(hidden_width) = hidden_depth
            activation (str): "relu" or "swish"
            batch_norm (bool)
            max_epochs_since_update (int): number of epochs to wait for improvement during training
        """
        output_dim = 2 * target_dim
        super().__init__(input_dim, output_dim, hidden_width,
                         hidden_depth, activation, batch_norm)
        self.max_epochs_since_update = max_epochs_since_update

        # initialize other parameters and buffers
        self.register_parameter("max_logvar", Parameter(torch.tensor([0.5])))
        self.register_parameter("min_logvar", Parameter(torch.tensor([-10.])))
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_std", torch.ones(input_dim))
        self.register_buffer("target_mean", torch.zeros(target_dim))
        self.register_buffer("target_std", torch.ones(target_dim))

        self._train_ckpt = deepcopy(self.state_dict())
        self._eval_ckpt = deepcopy(self.state_dict())

    def forward(self, inputs):
        """
        Args:
            inputs (torch.Tensor): [n x input_dim]
        Returns:
            mean (torch.Tensor): [n x target_dim]
            var (torch.Tensor): [n x target_dim]
        """
        assert torch.is_tensor(inputs) and inputs.dim() == 2
        inputs = (inputs - self.input_mean) / self.input_std
        output = super().forward(inputs)
        mean, logvar = output.chunk(2, dim=-1)

        logvar = self.max_logvar - softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + softplus(logvar - self.min_logvar)
        var = logvar.exp()
        mean = mean * self.target_std + self.target_mean
        var = var * self.target_std.pow(2)

        return mean, var

    def predict(self, np_inputs):
        """
        Args:
            np_inputs (np.array): [n x input_dim]
        Returns:
            mean (np.array): [n x target_dim]
            var (np.array): [n x target_dim]
        """
        inputs = torch.tensor(np_inputs, dtype=torch.get_default_dtype())
        with torch.no_grad():
            mean, var = self(inputs)
        return mean.cpu().numpy(), var.cpu().numpy()

    def fit(self, train_data, holdout_data, batch_size, lr, logvar_penalty_coeff,
            early_stopping=True, normalize=True, max_epochs=None, max_steps=None, verbose=False):
        """
        Args:
            train_data (tuple of nd.arrays): ([n_train x input_dim], [n_train x target_dim])
            holdout_data (tuple of nd.arrays): ([n_holdout x input_dim], [n_holdout x target_dim])
            batch_size (int)
            lr (float)
            logvar_penalty_coeff (float)
            early_stopping (bool): if True, terminate training once holdout loss stops improving
            normalize (bool): if True, z-score inputs and targets
            max_epochs (int): max number of epochs to train
            max_steps (int): max number of optimizer steps to take during training
            verbose (bool)
        """
        train_data = torch.utils.data.TensorDataset(
            torch.tensor(train_data[0], dtype=torch.get_default_dtype()),
            torch.tensor(train_data[1], dtype=torch.get_default_dtype())
        )
        holdout_data = torch.utils.data.TensorDataset(
            torch.tensor(holdout_data[0], dtype=torch.get_default_dtype()),
            torch.tensor(holdout_data[1], dtype=torch.get_default_dtype())
        )

        def obj_fn(pred_dist, targets):
            log_prob = pred_dist.log_prob(targets).mean()
            aux_loss = logvar_penalty_coeff * (self.max_logvar - self.min_logvar)
            return log_prob - aux_loss

        if holdout_data:
            val_x, val_y = holdout_data[:]
            eval_loss, eval_mse = self._get_val_metrics(obj_fn, torch.nn.MSELoss(), val_x, val_y)
            if verbose:
                print(f"[ ProbMLP ] initial holdout loss: {eval_loss:.4f}, MSE: {eval_mse:.4f}")
        else:
            eval_loss, eval_mse = 1e6, 1e6
        snapshot = (0, eval_loss)

        self.load_state_dict(self._train_ckpt)

        if normalize:
            train_inputs, train_targets = train_data[:]
            self.input_mean, self.input_std = train_inputs.mean(0), train_inputs.std(0)
            self.target_mean, self.target_std = train_targets.mean(0), train_targets.std(0)

        if verbose:
            print(f"[ ProbMLP ] training on {len(train_data)} examples")
        optimizer = Adam(self.optim_param_groups, lr=lr)
        metrics, snapshot = self._training_loop(train_data, holdout_data, batch_size,
                                                optimizer, obj_fn, snapshot, max_epochs,
                                                early_stopping, max_steps)

        self._train_ckpt = deepcopy(self.state_dict())
        self.load_state_dict(self._eval_ckpt)
        self.train()
        if holdout_data:
            eval_loss, eval_mse = self._get_val_metrics(obj_fn, torch.nn.MSELoss(), val_x, val_y)
            metrics['holdout_mse'] = eval_mse
            metrics['holdout_loss'] = eval_loss

            if verbose:
                print(f"[ ProbMLP ] holdout loss: {metrics['val_loss'][-1]:.4f}, MSE: {metrics['val_mse'][-1]:.4f}")
                print(f"[ ProbMLP ] loading snapshot from epoch {snapshot[0]}")
                print(f"[ ProbMLP ] final holdout loss: {eval_loss:.4f}, MSE: {eval_mse:.4f}")

        self.eval()
        return metrics

    def _training_loop(self, train_dataset, val_dataset, batch_size, optimizer,
                       obj_fn, snapshot, max_epochs, early_stopping, max_steps):
        metrics = {
            'train_loss': [],
            'val_loss': [],
            'val_mse': [],
        }
        exit_training = False
        num_batches = math.ceil(len(train_dataset) / batch_size)
        epoch = snapshot[0] + 1
        avg_train_loss = None
        alpha = 2 / (num_batches + 1)
        mse_fn = torch.nn.MSELoss()
        steps = 1
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        if val_dataset:
            val_x, val_y = val_dataset[:]

        while not exit_training:
            self.train()
            for inputs, labels in dataloader:
                if inputs.shape[0] <= 1:
                    continue
                optimizer.zero_grad()
                pred_mean, pred_var = self(inputs)
                pred_dist = Normal(pred_mean, pred_var.sqrt())
                loss = -obj_fn(pred_dist, labels)
                loss.backward()
                optimizer.step()

                if avg_train_loss:
                    avg_train_loss = alpha * loss.item() + (1 - alpha) * avg_train_loss
                else:
                    avg_train_loss = loss.item()

                if max_steps and steps == max_steps:
                    print("trained for max steps allowed, breaking")
                    return metrics, snapshot
                steps += 1

            if val_dataset:
                val_loss, val_mse = self._get_val_metrics(obj_fn, mse_fn, val_x, val_y)
                metrics['val_loss'].append(val_loss)
                metrics['val_mse'].append(val_mse)
            conv_metric = val_loss if early_stopping else avg_train_loss

            snapshot, exit_training = self.save_best(snapshot, epoch, conv_metric)
            epoch += 1
            metrics['train_loss'].append(avg_train_loss)
            if exit_training or (max_epochs and epoch == max_epochs):
                break

        return metrics, snapshot

    def _get_val_metrics(self, obj_fn, mse_fn, inputs, targets):
        with torch.no_grad():
            self.eval()
            pred_mean, pred_var = self(inputs)
            pred_dist = Normal(pred_mean, pred_var.sqrt())
            val_loss = -obj_fn(pred_dist, targets)
            val_mse = mse_fn(pred_mean, targets)
        return [val_loss.item(), val_mse.item()]

    def save_best(self, snapshot, epoch, current_loss):
        exit_training = False
        last_update, best_loss = snapshot
        improvement = (best_loss - current_loss) / abs(best_loss)
        if improvement > 0.001:
            snapshot = (epoch, current_loss)
            self._eval_ckpt = deepcopy(self.state_dict())
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
