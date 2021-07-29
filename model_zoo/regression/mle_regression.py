import numpy as np
import torch

from torch import nn
from copy import deepcopy

import model_zoo
from model_zoo.utils.training import save_best


class MaxLikelihoodRegression(torch.nn.Module):
    def __init__(self, input_dim, target_dim, model_class, model_kwargs, mode='prob'):
        super().__init__()
        self._deterministic = (mode == 'det')
        output_dim = 2 * target_dim
        model_constructor = getattr(model_zoo.architecture, model_class)
        self.model = model_constructor(input_dim, output_dim, **model_kwargs)

        self.register_parameter("max_logvar", nn.Parameter(torch.tensor((0.5))))
        self.register_parameter("min_logvar", nn.Parameter(torch.tensor((-10.))))

        self.register_buffer("input_mean", torch.zeros(input_dim, dtype=torch.get_default_dtype()))
        self.register_buffer("input_std", torch.ones(input_dim, dtype=torch.get_default_dtype()))
        self.register_buffer("target_mean", torch.zeros(target_dim, dtype=torch.get_default_dtype()))
        self.register_buffer("target_std", torch.ones(target_dim, dtype=torch.get_default_dtype()))

        self.train_checkpoint = deepcopy(self.state_dict())
        self.eval_checkpoint = deepcopy(self.state_dict())

    def forward(self, inputs):
        """
        Args:
            inputs (torch.Tensor): [n x input_dim]
        Returns:
            mean (torch.Tensor): [n x target_dim]
            var (torch.Tensor): [n x target_dim]
        """
        self._check_dim(inputs)
        inputs = (inputs - self.input_mean) / self.input_std
        output = self.model.forward(inputs)
        mean, logvar = output.chunk(2, dim=-1)
        logvar = self.max_logvar - nn.functional.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + nn.functional.softplus(logvar - self.min_logvar)
        mean = mean * self.target_std + self.target_mean
        var = logvar.exp() * self.target_std ** 2
        return mean, var

    def predict(self, inputs, compat_mode='np'):
        """
        Args:
            inputs (np.array): [num_batch x input_dim] or [num_batch x seq_len x input_dim]
            compat_mode: 'np' or 'torch'
        Returns:
            mean (np.array): [*batch_shape x target_dim]
            var (np.array): [*batch_shape x target_dim]
        """
        if torch.is_tensor(inputs):
            pass
        else:
            inputs = torch.tensor(inputs, dtype=torch.get_default_dtype())
        mean, var = self(inputs)

        if compat_mode == 'np':
            mean, var = mean.detach().cpu().numpy(), var.detach().cpu().numpy()
        elif compat_mode == 'torch':
            pass
        else:
            raise ValueError("unrecognized compatibility mode, use 'np' (NumPy) or 'torch' (PyTorch)")

        return mean, var

    def sample(self, np_inputs):
        mean, var = self.predict(np_inputs)
        noise = np.random.randn(np_inputs.shape)
        res = mean if self._deterministic else mean + np.sqrt(var) * noise
        return res

    def validate(self, np_inputs, np_targets):
        inputs = torch.tensor(np_inputs, dtype=torch.get_default_dtype())
        targets = torch.tensor(np_targets, dtype=torch.get_default_dtype())
        self.model.reset()
        with torch.no_grad():
            pred_mean, pred_var = self(inputs)
            mse = (pred_mean - targets).pow(2).mean()
            loss = self.loss_fn(inputs, targets, beta=1e-2)
        metrics = {'val_mse': mse.item(), 'val_loss': loss.item()}

        return metrics

    def fit(self, dataset, fit_params):
        """
        :param dataset (model_zoo.utils.data.Dataset)
        :param fit_params = {
                    lr=1e-3,
                    weight_decay=1e-4,
                    batch_size=32,
                    logvar_penalty_coeff=1e-2,
                    early_stopping=True,
                    wait_epochs=10,
                    wait_tol=1e-3,

            }
        :return: metrics dict
        """
        fit_params = dict(fit_params)

        val_data = dataset.holdout_data
        val_mse = self.validate(*val_data)["val_mse"]
        snapshot = (0, val_mse, self.eval_checkpoint)
        self.load_state_dict(self.train_checkpoint)

        normalize = fit_params.setdefault('normalize', True)
        if normalize:
            input_stats, target_stats = dataset.get_stats(compat_mode='torch')
            self.input_mean, self.input_std = input_stats
            self.target_mean, self.target_std = target_stats

        # main training loop
        train_loader = dataset.get_loader(fit_params['batch_size'])

        # adding this to the config causes issues with nested instantiation.
        # optimizer = hydra.utils.instantiate(fit_params['optimizer'], params=self._optim_p_groups)
        optimizer = torch.optim.Adam(self._optim_p_groups, lr=fit_params["lr"], weight_decay=fit_params['weight_decay'])

        snapshot, train_metrics = self._training_loop(train_loader, optimizer,
                                       val_data, snapshot, fit_params)

        self.train_checkpoint = deepcopy(self.state_dict())
        _, _, self.eval_checkpoint = snapshot
        self.load_state_dict(self.eval_checkpoint)
        self.eval()
        fit_metrics = self.validate(*val_data)
        fit_metrics.update(train_metrics)
        return fit_metrics

    def loss_fn(self, inputs, targets, beta):
        self._check_dim(inputs)
        self.model.reset()
        pred_mean, pred_var = self(inputs)
        likelihood = self.likelihood(pred_mean, pred_var, targets)
        mse = (pred_mean - targets).pow(2).mean()
        var_bound_loss = beta * (self.max_logvar - self.min_logvar)
        loss = mse if self._deterministic else (-likelihood + var_bound_loss)
        return loss

    def likelihood(self, pred_mean, pred_var, targets):
        target_loss = (pred_mean - targets).pow(2).div(pred_var).mean()
        var_loss = pred_var.log().mean()
        return -target_loss - var_loss

    def _training_loop(self, train_loader, optimizer, holdout_data, snapshot, fit_params):
        metrics = {'train_loss': [], 'val_loss': []}
        num_batches = len(train_loader)
        alpha = 2 / (num_batches + 1)  # exp. moving average parameter
        num_updates = 0
        train_loss = None

        early_stopping = fit_params.setdefault('early_stopping', False)
        wait_epochs = fit_params.setdefault('wait_epochs', None)
        wait_tol = fit_params.setdefault('wait_tol', None)
        max_grad_norm = fit_params.setdefault('max_grad_norm', None)
        max_updates = fit_params.setdefault('max_updates', None)

        exit_training = False
        epoch, _, _ = snapshot
        while not exit_training:
            self.train()
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                loss = self.loss_fn(inputs, targets, fit_params['logvar_penalty_coeff'])
                loss.backward()
                if max_grad_norm:
                    self._clip_grads(optimizer, max_grad_norm)
                optimizer.step()
                train_loss = loss.item() if train_loss is None else ((1 - alpha) * train_loss + alpha * loss.item())

                num_updates += 1
                if max_updates is not None and num_updates == max_updates:
                    exit_training = True
                    break

            self.eval()
            with torch.no_grad():
                val_metrics = self.validate(*holdout_data)
            conv_metric = val_metrics['val_mse'] if early_stopping else train_loss
            converged, snapshot = save_best(self, conv_metric, epoch, snapshot, wait_epochs, wait_tol)
            exit_training = converged if converged else exit_training

            metrics['train_loss'].append(train_loss)
            metrics['val_loss'].append(val_metrics['val_loss'])
            epoch += 1

        return snapshot, metrics

    def _clip_grads(self, optimizer, max_grad_norm):
        assert len(optimizer.param_groups) == 1
        params = optimizer.param_groups[0]['params']
        torch.nn.utils.clip_grad_norm_(params, max_grad_norm)

    def _check_dim(self, inputs):
        if torch.is_tensor(inputs):
            if inputs.dim() < 2 or inputs.dim() > 3:
                raise ValueError('2D or 3D inputs expected')
        elif isinstance(inputs, np.ndarray):
            if inputs.ndim < 2 or inputs.ndim > 3:
                raise ValueError('2D or 3D inputs expected')

    @property
    def _optim_p_groups(self):
        return self.parameters()

    def reset(self):
        self.model.reset()
