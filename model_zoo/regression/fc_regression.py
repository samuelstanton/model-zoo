import math
import torch

from copy import deepcopy

from torch.nn import Linear, Parameter
from torch.nn.functional import softplus
from torch.distributions import Normal
from torch.optim import Adam

from model_zoo.architecture import FCNet
from model_zoo.utils.data import SeqDataset
from model_zoo.utils.training import save_best


class FCRegression(FCNet):
    """ Fully-connected neural network regression model w/ Gaussian predictive distributions

    Behavior should closely mimic that of the component networks in Kurtland Chua's
    bootstrapped deep ensemble implementation (https://tinyurl.com/vl4alu9)
    """

    def __init__(self, input_dim, target_dim, hidden_width, hidden_depth=4,
                 activation="relu", batch_norm=True, init="default", max_epochs_since_update=5) -> None:
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
                         hidden_depth, activation, batch_norm, init)
        self.max_epochs_since_update = max_epochs_since_update

        # initialize other parameters and buffers
        self.register_parameter("max_logvar", Parameter(torch.tensor((0.5))))
        self.register_parameter("min_logvar", Parameter(torch.tensor((-10.))))
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

    def fit(self, dataset, fit_params, verbose=False):
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
        if isinstance(dataset, SeqDataset):
            dataset.subseq_format('flat')
        train_loader = dataset.get_loader(fit_params['batch_size'])
        holdout_data = [torch.tensor(array, dtype=torch.get_default_dtype()) for array in dataset.get_holdout_data()]

        def loss_fn(pred_dist, targets):
            target_loss = (pred_dist.mean - targets).pow(2).div(pred_dist.variance).mean()
            var_loss = pred_dist.variance.log().mean()
            beta = fit_params['logvar_penalty_coeff']
            var_bound_loss = beta * (self.max_logvar - self.min_logvar)
            return target_loss + var_loss + var_bound_loss

        if holdout_data:
            h_inputs, h_targets = holdout_data
            eval_loss, eval_mse = self._get_val_metrics(loss_fn, torch.nn.MSELoss(), h_inputs, h_targets)
            if verbose:
                print(f"[ ProbMLP ] initial holdout loss: {eval_loss:.4f}, MSE: {eval_mse:.4f}")
        else:
            eval_loss, eval_mse = 1e6, 1e6
        snapshot = (0, eval_mse, self._eval_ckpt)
        self.load_state_dict(self._train_ckpt)

        # TODO fix this
        # if fit_params['normalize']:
        #     train_inputs, train_targets = dataset[:]
        #     self.input_mean, self.input_std = train_inputs.mean(0), train_inputs.std(0)
        #     self.target_mean, self.target_std = train_targets.mean(0), train_targets.std(0)

        if verbose:
            print(f"[ ProbMLP ] training on {len(dataset)} example sequences")
        optimizer = Adam(self.optim_param_groups, lr=fit_params['lr'])
        metrics, snapshot = self._training_loop(train_loader, optimizer, loss_fn, holdout_data, snapshot, fit_params)

        self._train_ckpt = deepcopy(self.state_dict())
        _, _, self._eval_ckpt = snapshot
        self.load_state_dict(self._eval_ckpt)
        self.train()
        if holdout_data:
            eval_loss, eval_mse = self._get_val_metrics(loss_fn, torch.nn.MSELoss(), h_inputs, h_targets)
            metrics['holdout_mse'] = eval_mse
            metrics['holdout_loss'] = eval_loss

            if verbose:
                print(f"[ ProbMLP ] holdout loss: {metrics['val_loss'][-1]:.4f}, MSE: {metrics['val_mse'][-1]:.4f}")
                print(f"[ ProbMLP ] loading snapshot from epoch {snapshot[0]}")
                print(f"[ ProbMLP ] final holdout loss: {eval_loss:.4f}, MSE: {eval_mse:.4f}")

        self.eval()
        return metrics

    def _training_loop(self, train_loader, optimizer, loss_fn, holdout_data, snapshot, fit_params):
        metrics = {
            'train_loss': [],
            'val_loss': [],
            'val_mse': [],
        }
        if holdout_data:
            h_inputs, h_targets = holdout_data

        num_batches = len(train_loader)
        alpha = 2 / (num_batches + 1)
        avg_train_loss = None
        mse_fn = torch.nn.MSELoss()

        fit_params = dict(fit_params)
        max_steps = fit_params.setdefault('max_steps', None)
        max_epochs = fit_params.setdefault('max_epochs', None)
        early_stopping = fit_params.setdefault('early_stopping', False)
        wait_epochs = fit_params['wait_epochs']
        wait_tol = fit_params['wait_tol']

        steps = 1
        epoch = snapshot[0]
        exit_training = False
        while not exit_training:
            self.train()
            for inputs, labels in train_loader:
                if inputs.shape[0] <= 1:
                    continue
                optimizer.zero_grad()
                pred_mean, pred_var = self(inputs)
                pred_dist = Normal(pred_mean, pred_var.sqrt())
                loss = loss_fn(pred_dist, labels)
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

            if holdout_data:
                val_loss, val_mse = self._get_val_metrics(loss_fn, mse_fn, h_inputs, h_targets)
                metrics['val_loss'].append(val_loss)
                metrics['val_mse'].append(val_mse)
            conv_metric = val_mse if early_stopping else avg_train_loss

            epoch += 1
            exit_training, snapshot = save_best(self, conv_metric, epoch, snapshot, wait_epochs, wait_tol)
            metrics['train_loss'].append(avg_train_loss)
            if exit_training or (max_epochs and epoch == max_epochs):
                break

        return metrics, snapshot

    def _get_val_metrics(self, loss_fn, mse_fn, inputs, targets):
        with torch.no_grad():
            self.eval()
            pred_mean, pred_var = self(inputs)
            pred_dist = Normal(pred_mean, pred_var.sqrt())
            val_loss = loss_fn(pred_dist, targets)
            val_mse = mse_fn(pred_mean, targets)
        return [val_loss.item(), val_mse.item()]

    # def save_best(self, snapshot, epoch, current_loss):
    #     exit_training = False
    #     last_update, best_loss = snapshot
    #     improvement = (best_loss - current_loss) / abs(best_loss)
    #     if improvement > 0.001:
    #         snapshot = (epoch, current_loss)
    #         self._eval_ckpt = deepcopy(self.state_dict())
    #     if epoch == snapshot[0] + self.max_epochs_since_update:
    #         exit_training = True
    #     return snapshot, exit_training

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
