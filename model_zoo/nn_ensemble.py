import copy
import math
import torch
import numpy as np

from torch import Tensor
from torch.nn import Module, ModuleList, Parameter
from torch.nn.functional import softplus
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.utils.data import TensorDataset, DataLoader
from .fc import FC
from collections import OrderedDict


class PytorchBNN(Module):
    """
    Ensemble of probabilistic neural networks.
    Should closely replicate the behavior of mbpo.models.bnn.BNN
    """
    def __init__(
            self,
            input_shape: torch.Size,
            label_shape: torch.Size,
            hidden_width: int,
            hidden_depth: int,
            ensemble_size: int,
            minibatch_size: int,
            lr: float,
            logvar_penalty_coeff: float,
            max_epochs_since_update: int,
            **kwargs
    ):
        super().__init__()
        self.input_shape = input_shape
        self.label_shape = label_shape
        self.output_shape = label_shape[:-1]
        self.output_shape += torch.Size([2*label_shape[-1]])
        self._hidden_width = hidden_width
        self._hidden_depth = hidden_depth
        self._ensemble_size = ensemble_size
        self._minibatch_size = minibatch_size
        self._lr = lr
        self._logvar_penalty_coeff = logvar_penalty_coeff
        self._snapshots = [(0, 1e6)] * ensemble_size
        self._epochs_since_update = 0
        self._max_epochs_since_update = max_epochs_since_update

        components = [
            FC(
                input_shape,
                self.output_shape,
                hidden_width,
                hidden_depth,
                activation='swish',
                batch_norm=False
            ) for _ in range(ensemble_size)
        ]
        self.nn_components = ModuleList(components)

        for idx in range(ensemble_size):
            network = self.nn_components[idx]
            network.register_parameter(
                f"max_logvar", Parameter(torch.ones(*label_shape) / 2.)
            )
            network.register_parameter(
                f"min_logvar", Parameter(-torch.ones(*label_shape) * 10.)
            )

        # initialize preprocessers
        self.register_buffer(
            "input_mean", torch.zeros(*input_shape)
        )
        self.register_buffer(
            "input_std", torch.ones(*input_shape)
        )
        self.register_buffer(
            "label_mean", torch.zeros(*label_shape)
        )
        self.register_buffer("label_std", torch.ones(*label_shape))

    def _train(self):
        Module.train(self)

    def eval(self):
        return Module.train(self, False)

    def random_inds(self, batch_size):
        inds = np.random.choice(list(range(self._ensemble_size)), size=batch_size)
        return inds

    def sample_next(self, inputs):
        with torch.no_grad():
            pred_dist = self.predict(inputs)
        return pred_dist.sample().t().cpu().numpy()

    def forward(self, inputs: Tensor, idx: int):
        # flatten, whiten inputs
        if inputs.dim() > 2:
            inputs = inputs.flatten(end_dim=-2)
        inputs = (inputs - self.input_mean) / self.input_std
        network = self.nn_components[idx]
        # predict mean, variance
        output = network(inputs)
        mean, logvar = output.view(-1, *self.output_shape).chunk(2, dim=-1)
        max_logvar = getattr(network, f"max_logvar")
        min_logvar = getattr(network, f"min_logvar")
        logvar = max_logvar - softplus(max_logvar - logvar)
        logvar = min_logvar + softplus(logvar - min_logvar)
        std = logvar.exp().sqrt()
        # unwhiten predictions
        mean = mean * self.label_std + self.label_mean
        mean = mean.permute(*range(1, len(self.label_shape) + 1), 0)
        std = std * self.label_std
        std = std.permute(*range(1, len(self.label_shape) + 1), 0)
        return Normal(mean, std)

    def predict(self, inputs, factored=False, **kwargs):
        # n = inputs.shape[0]
        inputs = torch.tensor(inputs, dtype=torch.get_default_dtype())
        if inputs.dim() == 2 and len(self.input_shape) > 1:
            inputs = inputs.t().expand(*self.input_shape, -1)
            inputs = inputs.permute(-1, *range(len(self.input_shape)))
        pred_dists = []
        for idx in range(len(self.nn_components)):
            with torch.no_grad():
                pred_dists.append(self.forward(inputs, idx))
        pred_means = np.stack([dist.mean.t().cpu().numpy() for dist in pred_dists])
        pred_vars = np.stack([dist.variance.t().cpu().numpy() for dist in pred_dists])

        if factored:
            return pred_means, pred_vars
        else:
            raise NotImplementedError

    def fit(
            self,
            inputs: np.ndarray,
            targets: np.ndarray,
            holdout_ratio=0.2,
            max_epochs=None,
            max_kl=None,
            normalize=True,
            verbose=False,
            **kwargs
        ):
        metrics = {
            'avg_train_nll': [],
            'holdout_nll': [],
            'holdout_mse': [],
        }
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(inputs, dtype=torch.get_default_dtype()),
            torch.tensor(targets, dtype=torch.get_default_dtype())
        )
        n_val = min(int(5e3), int(holdout_ratio * len(dataset)))
        n_train = len(dataset) - n_val
        train_data, val_data = torch.utils.data.random_split(dataset, [n_train, n_val])
        train_x, train_y = train_data[:]
        val_x, val_y = val_data[:]

        if normalize:
            self.input_mean = train_x.mean(0)
            self.input_std = train_x.std(0)

        bootstraps = []
        for _ in range(len(self.nn_components)):
            weights = torch.ones(len(train_data)) / len(train_data)
            bootstrap_idx = torch.multinomial(weights, len(train_data), replacement=True)
            bootstraps.append(TensorDataset(train_x[bootstrap_idx], train_y[bootstrap_idx]))

        # set up train metrics
        first_batch_flag = [1] * len(self.nn_components)
        avg_batch_nll = torch.empty(len(self.nn_components))
        num_batches = math.ceil(len(train_data) / self._minibatch_size)
        alpha = 2 / (num_batches + 1)

        # set up validation metrics
        mse_loss = torch.nn.MSELoss()
        val_mse = torch.empty(len(self.nn_components))
        val_nll = torch.empty(len(self.nn_components))
        val_kl = torch.empty(len(self.nn_components))

        ref_val_pred = []
        for idx in range(len(self.nn_components)):
            with torch.no_grad():
                self.eval()
                ref_val_pred.append(self.forward(val_x, idx))

        # set up training loop
        epoch = 0
        self._epochs_since_update = 0
        self._snapshots = [(0, 1e6)] * self._ensemble_size
        best_weights = [network.state_dict() for network in self.nn_components]

        exit_training = False
        while not exit_training:
            self._train()
            if max_epochs and epoch == max_epochs:
                print("trained for max allowed number of epochs")
                break
            for idx, network in enumerate(self.nn_components):
                dataloader = DataLoader(bootstraps[idx], batch_size=self._minibatch_size, shuffle=True)
                max_logvar = getattr(network, f"max_logvar")
                min_logvar = getattr(network, f"min_logvar")
                optimizer = torch.optim.Adam(network.optim_param_groups, lr=self._lr)

                for inputs, targets in dataloader:
                    optimizer.zero_grad()
                    pred_dist = self.forward(inputs, idx)
                    likelihood = pred_dist.log_prob(targets.t()).mean()
                    penalty_coeff = self._logvar_penalty_coeff
                    loss = -likelihood + penalty_coeff * (max_logvar - min_logvar).sum()
                    loss.backward()
                    optimizer.step()

                    if first_batch_flag[idx] == 1:
                        avg_batch_nll[idx] = -likelihood.detach()
                        first_batch_flag[idx] = 0
                    else:
                        avg_batch_nll[idx] = alpha * (-likelihood).detach() + (1 - alpha) * avg_batch_nll[idx]

                with torch.no_grad():
                    val_pred = self.forward(val_x, idx)
                    val_kl[idx] = kl_divergence(val_pred, ref_val_pred[idx]).mean()
                    val_nll[idx] = -val_pred.log_prob(val_y.t()).mean()
                    val_mse[idx] = mse_loss(val_pred.mean.t(), val_y)

            best_weights, exit_training = self._save_best(epoch, val_mse, best_weights)
            metrics['avg_train_nll'].append(avg_batch_nll.clone())
            metrics['holdout_mse'].append(val_mse.clone())
            metrics['holdout_nll'].append(val_nll.clone())
            epoch += 1

            if verbose or exit_training:
                print(f"epoch: {epoch:0>3d} - avg holdout MSE: {round(val_mse.mean().item(), 4)}")
            if exit_training:
                print(f"training converged")
            if max_kl and val_kl.max() > max_kl:
                print(f"holdout KL {val_kl.max().item():.2f} exceeds threshold. Halting after {epoch} epochs")
                break

        # reload weights with best holdout MSE
        for idx, network in enumerate(self.nn_components):
            network.load_state_dict(best_weights[idx])
        self.eval()
        model_metrics = {'val_loss': val_mse.mean().cpu().numpy()}
        return OrderedDict(model_metrics)

    def _save_best(self, epoch, holdout_losses, best_weights):
        updated = False
        exit_training = False
        for i in range(len(holdout_losses)):
            current = holdout_losses[i]
            _, best = self._snapshots[i]
            improvement = (best - current) / abs(best)
            if improvement > 0.01:
                self._snapshots[i] = (epoch, current.item())
                best_weights[i] = self.nn_components[i].state_dict()
                updated = True

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1

        if self._epochs_since_update >= self._max_epochs_since_update:
            exit_training = True

        return best_weights, exit_training

    def get_batch_model(self, new_batch_dims):
        new_input_shape = torch.Size(list(new_batch_dims) + list(self.input_shape))
        new_output_shape = torch.Size(list(new_batch_dims) + list(self.output_shape))
        new_label_shape = torch.Size(list(new_batch_dims) + list(self.label_shape))
        new_model = copy.deepcopy(self)
        new_model.input_shape = new_input_shape
        new_model.output_shape = new_output_shape
        new_model.label_shape = new_label_shape
        return new_model

    def make_copy(self):
        return copy.deepcopy(self)
