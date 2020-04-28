import numpy as np
import torch

from torch import nn
from copy import deepcopy
from model_zoo.utils.training import save_best


class MaxLikelihoodRegression(torch.nn.Module):
    def __init__(self, input_dim, target_dim, model_class, model_kwargs, deterministic=False):
        super().__init__()
        self._deterministic = deterministic
        output_dim = 2 * target_dim
        self.model = model_class(input_dim, output_dim, **model_kwargs)

        self.register_parameter("max_logvar", nn.Parameter(torch.tensor((0.5))))
        self.register_parameter("min_logvar", nn.Parameter(torch.tensor((-10.))))

        self.input_stat_tracker = nn.BatchNorm1d(input_dim, affine=False)
        self.target_stat_tracker = nn.BatchNorm1d(target_dim, affine=False)

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
        assert torch.is_tensor(inputs) and inputs.dim() == 2
        inputs = self.input_stat_tracker(inputs)
        output = self.model.forward(inputs)
        mean, logvar = output.chunk(2, dim=-1)
        logvar = self.max_logvar - nn.functional.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + nn.functional.softplus(logvar - self.min_logvar)
        # mean = mean * self.target_var.sqrt() + self.target_mean
        # var = logvar.exp() * self.target_var
        var = logvar.exp()
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

    def sample(self, np_inputs):
        mean, var = self.predict(np_inputs)
        noise = np.random.randn(np_inputs.shape)
        res = mean if self._deterministic else mean + np.sqrt(var) * noise
        return res

    def validate(self, np_inputs, np_targets):
        self._check_dim(np_inputs)
        self.model.reset()
        if np_inputs.ndim == 2:
            mean, var = self.predict(np_inputs)
            mse = np.power(mean - np_targets, 2).mean()
        else:
            _, seq_len, _ = np_inputs.shape
            mse = 0
            for i in range(seq_len):
                mean, var = self.predict(np_inputs[:, i])
                mse += np.power(mean - np_targets[:, i], 2).mean() / seq_len

        inputs = torch.tensor(np_inputs, dtype=torch.get_default_dtype())
        targets = torch.tensor(np_targets, dtype=torch.get_default_dtype())
        val_loss = self.loss_fn(inputs, targets, beta=1e-2)
        metrics = {'val_mse': mse.item(), 'val_loss': val_loss.item()}

        return metrics

    def fit(self, dataset, fit_params):
        """
        :param dataset (model_zoo.utils.data.SeqDataset)
        :param fit_params (dict)
        :return:
        """
        train_loader = dataset.get_loader(fit_params['batch_size'])
        optimizer = torch.optim.Adam(self._optim_p_groups, lr=fit_params["lr"])

        holdout_data = dataset.get_holdout_data()
        holdout_mse = self.validate(*holdout_data)["val_mse"]
        snapshot = (0, holdout_mse, self.eval_checkpoint)
        self.load_state_dict(self.train_checkpoint)

        # main training loop
        snapshot, train_metrics = self._training_loop(train_loader, optimizer,
                                       holdout_data, snapshot, fit_params)

        self.train_checkpoint = deepcopy(self.state_dict())
        _, holdout_loss, self.eval_checkpoint = snapshot
        fit_metrics = {"holdout_mse": holdout_loss}
        fit_metrics.update(train_metrics)
        self.load_state_dict(self.eval_checkpoint)
        self.eval()
        return fit_metrics

    def loss_fn(self, inputs, targets, beta):
        self._check_dim(inputs)
        self.model.reset()
        if inputs.dim() == 2:
            mean, var = self(inputs)
            likelihood = self.likelihood(mean, var, targets)
            mse = (mean - targets).pow(2).mean()

        else:
            n, seq_len, input_dim = inputs.shape
            means, vars = [], []
            for t in range(seq_len):
                curr_input = inputs[:, t]
                # curr_input = self.input_stat_tracker(inputs[:, t])
                mean, var = self(curr_input)
                means.append(mean)
                vars.append(var)
                # with torch.no_grad():
                #     targets[:, t] = self.target_stat_tracker(targets[:, t])
            means = torch.stack(means, dim=-2)
            vars = torch.stack(vars, dim=-2)
            likelihood = self.likelihood(means, vars, targets)
            mse = (means - targets).pow(2).mean()

        var_bound_loss = beta * (self.max_logvar - self.min_logvar)
        loss = mse if self._deterministic else (-likelihood + var_bound_loss)
        return loss

    def likelihood(self, pred_mean, pred_var, targets):
        target_loss = (pred_mean - targets).pow(2).div(pred_var).mean()
        var_loss = pred_var.log().mean()
        return -target_loss - var_loss

    def _training_loop(self, train_loader, optimizer, holdout_data, snapshot, fit_params):
        metrics = {'train_loss': [], 'holdout_loss': []}
        fit_params = dict(fit_params)

        num_batches = len(train_loader)
        alpha = 2 / (num_batches + 1)  # exp. moving average parameter
        train_loss = None

        wait_epochs = fit_params.setdefault('wait_epochs', None)
        wait_tol = fit_params.setdefault('wait_tol', None)
        max_grad_norm = fit_params.setdefault('max_grad_norm', None)

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

            self.eval()
            holdout_metrics = self.validate(*holdout_data)
            exit_training, snapshot = save_best(self, holdout_metrics['val_loss'], epoch, snapshot, wait_epochs, wait_tol)
            metrics['train_loss'].append(train_loss)
            metrics['holdout_loss'].append(holdout_metrics['val_loss'])
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

    @property
    def input_mean(self):
        return self.input_stat_tracker.running_mean

    @property
    def input_var(self):
        return self.input_stat_tracker.running_var

    @property
    def target_mean(self):
        return self.target_stat_tracker.running_mean

    @property
    def target_var(self):
        return self.target_stat_tracker.running_var
