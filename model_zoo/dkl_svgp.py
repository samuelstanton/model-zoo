import math
import numpy as np
import torch

from copy import deepcopy

from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import GreaterThan
from gpytorch.models import GP
from gpytorch.variational import VariationalStrategy, MeanFieldVariationalDistribution
from gpytorch.mlls import VariationalELBO, PredictiveLogLikelihood
from .fc import FC
from sklearn.cluster import MiniBatchKMeans


class DeepFeatureSVGP(GP):
    def __init__(
            self,
            input_dim: int,
            feature_dim: int,
            label_dim: int,
            hidden_width: int or list,
            hidden_depth: int,
            n_inducing: int,
            batch_size: int,
            max_epochs_since_update,
            **kwargs
    ):
        params = locals()
        del params['self']
        self.__dict__ = params
        super().__init__()

        noise_constraint = GreaterThan(1e-4)
        self.likelihood = GaussianLikelihood(
            batch_shape=torch.Size([label_dim]),
            noise_constraint=noise_constraint
        )

        self.nn = FC(
            input_shape=torch.Size([input_dim]),
            output_shape=torch.Size([feature_dim]),
            hidden_width=hidden_width,
            depth=hidden_depth,
            batch_norm=True
        )
        self.batch_norm = torch.nn.BatchNorm1d(feature_dim)

        self.mean_module = ConstantMean(batch_shape=torch.Size([label_dim]))
        base_kernel = RBFKernel(
            batch_shape=torch.Size([label_dim]),
            ard_num_dims=feature_dim
        )
        self.covar_module = ScaleKernel(base_kernel, batch_shape=torch.Size([label_dim]))

        variational_dist = MeanFieldVariationalDistribution(
            num_inducing_points=n_inducing,
            batch_shape=torch.Size([label_dim])
        )
        inducing_points = torch.randn(n_inducing, feature_dim)
        self.variational_strategy = VariationalStrategy(
            self, inducing_points, variational_dist, learn_inducing_locations=True
        )

        # initialize preprocessers
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_std", torch.ones(input_dim))
        self.register_buffer("label_mean", torch.zeros(label_dim))
        self.register_buffer("label_std", torch.ones(label_dim))

        self._train_ckpt = deepcopy(self.state_dict())
        self._eval_ckpt = deepcopy(self.state_dict())

    def forward(self, features):
        mean = self.mean_module(features)
        covar = self.covar_module(features)
        return MultivariateNormal(mean, covar)

    def __call__(self, inputs):
        features = (inputs - self.input_mean) / self.input_std
        features = self.nn(features)
        features = self.batch_norm(features)
        features = features.expand(self.label_dim, -1, -1)

        return self.variational_strategy(features)

    def _predict_full(self, torch_inputs):
        with torch.no_grad():
            pred_dist = self(torch_inputs)
        mean = pred_dist.mean * self.label_std.view(self.label_dim, 1) + self.label_mean.view(self.label_dim, 1)
        covar = pred_dist.lazy_covariance_matrix * self.label_std.pow(2).view(self.label_dim, 1, 1)
        return MultivariateNormal(mean, covar)

    def predict(self, np_inputs, latent=False):
        inputs = torch.tensor(np_inputs, dtype=torch.get_default_dtype())
        with torch.no_grad():
            pred_dist = self(inputs) if latent else self.likelihood(self(inputs))
        mean = pred_dist.mean * self.label_std.view(self.label_dim, 1) + self.label_mean.view(self.label_dim, 1)
        var = pred_dist.variance * self.label_std.pow(2).view(self.label_dim, 1)
        return mean.t().cpu().numpy(), var.t().cpu().numpy()

    def fit(
        self,
        train_data,
        holdout_data,
        objective='elbo',
        max_epochs: int = None,
        normalize: bool = True,
        early_stopping: bool = False,
        pretrain: bool = False,
        reinit_inducing_loc: bool = False,
        verbose=False,
        **kwargs
    ):
        """
        Train the model on `dataset` by maximizing either the `VariationalELBO` or `PredictiveLogLikelihood` objective.
        :param dataset: `torch.utils.data.Dataset`
        Optional Arguments
        :param objective: `'elbo'` or `'pll'`.
        :param max_epochs: max number of epochs to train.
        :param holdout_ratio: proportion of `dataset` to hold out.
        :param normalize: If `True` normalize training inputs and labels
        :param early_stopping: If `True`, use holdout loss as convergence criterion.
                               Requires holdout_ratio > 0.
        :param pretrain: If `True`, pretrain the feature extractor with the MSE objective.
                         Requires self.feature_dim == self.label_dim.
        :param reinit_inducing_loc: If `True`, initialize inducing points with k-means.
        :return: metrics: `dict` with keys 'train_loss', 'val_loss', 'val_mse'.
        """
        train_data = torch.utils.data.TensorDataset(
            torch.tensor(train_data[0], dtype=torch.get_default_dtype()),
            torch.tensor(train_data[1], dtype=torch.get_default_dtype())
        )
        holdout_data = torch.utils.data.TensorDataset(
            torch.tensor(holdout_data[0], dtype=torch.get_default_dtype()),
            torch.tensor(holdout_data[1], dtype=torch.get_default_dtype())
        )

        if objective == 'elbo':
            obj_fn = VariationalELBO(self.likelihood, self, num_data=len(train_data))
        elif objective == 'pll':
            obj_fn = PredictiveLogLikelihood(self.likelihood, self, num_data=len(train_data), beta=1e-3)
        else:
            raise RuntimeError("unrecognized model objective")

        if holdout_data and early_stopping:
            val_x, val_y = holdout_data[:]
            eval_loss, eval_mse = self._get_val_metrics(obj_fn, torch.nn.MSELoss(), val_x, val_y)
        if eval_loss != eval_loss or not early_stopping:
            snapshot_loss = 1e6
        else:
            snapshot_loss = eval_loss
        snapshot = (0, snapshot_loss)

        if verbose:
            print(f"[ SVGP ] initial holdout loss: {eval_loss:.4f}, MSE: {eval_mse:.4f}")
        self.load_state_dict(self._train_ckpt)

        if normalize:
            train_inputs, train_labels = train_data[:]
            self.input_mean, self.input_std = train_inputs.mean(0), train_inputs.std(0)
            self.label_mean, self.label_std = train_labels.mean(0), train_labels.std(0)
            train_data = TensorDataset(
                train_inputs,
                (train_labels - self.label_mean) / self.label_std
            )

        if pretrain:
            if self.feature_dim == self.label_dim:
                if verbose:
                    print("[ SVGP ] pretraining feature extractor")
                self.nn.fit(
                    dataset=train_data,
                    holdout_ratio=0.,
                    early_stopping=False,
                )
            else:
                raise RuntimeError("features and labels must be the same size to pretrain")

        if reinit_inducing_loc:
            if verbose:
                print("[ SVGP ] initializing inducing point locations w/ k-means")
            train_inputs, _ = train_data[:]
            self.set_inducing_loc(train_inputs)

        if verbose:
            print(f"[ SVGP ] training w/ objective {objective} on {len(train_data)} examples")
        optimizer = Adam(self.optim_param_groups)
        if reinit_inducing_loc:
            temp = self.max_epochs_since_update
            self.max_epochs_since_update = 8
            loop_metrics, snapshot = self._training_loop(
                train_data,
                holdout_data,
                optimizer,
                obj_fn,
                snapshot,
                max_epochs,
                early_stopping
            )
            metrics = loop_metrics
            self.max_epochs_since_update = temp
            if verbose:
                print("[ SVGP ] dropping learning rate")

        for group in optimizer.param_groups:
            group['lr'] /= 10
        loop_metrics, snapshot = self._training_loop(
            train_data,
            holdout_data,
            optimizer,
            obj_fn,
            snapshot,
            max_epochs,
            early_stopping
        )
        if reinit_inducing_loc:
            for key in metrics.keys():
                metrics[key] += (loop_metrics[key])
        else:
            metrics = loop_metrics

        self._train_ckpt = deepcopy(self.state_dict())
        self.load_state_dict(self._eval_ckpt)
        self.train() # TODO investigate GPyTorch load_state_dict bug
        eval_loss, eval_mse = self._get_val_metrics(obj_fn, torch.nn.MSELoss(), val_x, val_y)
        metrics['holdout_mse'] = eval_mse
        metrics['holdout_loss'] = eval_loss

        if verbose:
            print(f"[ SVGP ] holdout loss: {metrics['val_loss'][-1]:.4f}, MSE: {metrics['val_mse'][-1]:.4f}")
            print(f"[ SVGP ] loading snapshot from epoch {snapshot[0]}")
            print(f"[ SVGP ] final holdout loss: {eval_loss:.4f}, MSE: {eval_mse:.4f}")

        self.eval()
        return metrics

    def _training_loop(
            self,
            train_dataset,
            val_dataset,
            optimizer,
            obj_fn,
            snapshot,
            max_epochs,
            early_stopping
    ):
        metrics = {
            'train_loss': [],
            'val_loss': [],
            'val_mse': [],
        }
        exit_training = False
        num_batches = math.ceil(len(train_dataset) / self.batch_size)
        epoch = snapshot[0] + 1
        avg_train_loss = None
        alpha = 2 / (num_batches + 1)
        mse_fn = torch.nn.MSELoss()
        dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        if val_dataset:
            val_x, val_y = val_dataset[:]

        while not exit_training:
            self.train()
            for inputs, labels in dataloader:
                if inputs.shape[0] <= 1:
                    continue
                optimizer.zero_grad()
                out = self(inputs)
                loss = -obj_fn(out, labels.t()).sum()
                loss.backward()
                optimizer.step()

                if avg_train_loss:
                    avg_train_loss = alpha * loss.item() + (1 - alpha) * avg_train_loss
                else:
                    avg_train_loss = loss.item()

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

    def _get_val_metrics(self, obj_fn, mse_fn, val_x, val_y):
        with torch.no_grad():
            self.eval()
            val_pred = self._predict_full(val_x)
            val_mean = val_pred.mean
            # val_std = val_pred.variance.sqrt()
            # val_pred = torch.distributions.Normal(val_mean, val_std)
            # val_loss = -val_pred.log_prob(val_y.t()).mean()
            val_loss = -obj_fn(val_pred, val_y.t()).sum()
            val_mse = mse_fn(val_mean, val_y.t())
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

    def set_inducing_loc(self, train_inputs):
        self.eval()
        train_inputs = (train_inputs - self.input_mean) / self.input_std
        np_inputs = train_inputs.cpu().numpy()
        kmeans = MiniBatchKMeans(
            n_clusters=self.n_inducing,
            compute_labels=False,
            init_size=4 * self.n_inducing
        )
        kmeans.fit(np_inputs)
        centroids = torch.tensor(
            kmeans.cluster_centers_,
            dtype=torch.get_default_dtype()
        )
        centroids = self.batch_norm(self.nn(centroids))
        self.variational_strategy.inducing_points.data.copy_(centroids)

    def make_copy(self):
        new_model = DeepFeatureSVGP(**self.__dict__)
        new_model.load_state_dict(self.state_dict())
        return new_model

    @property
    def optim_param_groups(self):
        gp_params, nn_params, other_params = [], [], []
        for name, param in self.named_parameters():
            if 'nn' in name:
                nn_params.append(param)
            elif 'raw' in name:
                gp_params.append(param)
            else:
                other_params.append(param)

        gp_params = {
            'params': gp_params,
            'lr': 1e-2
        }
        other_params = {
            'params': other_params,
            'lr': 1e-2
        }
        nn_params = {
            'params': nn_params,
            'lr': 1e-3,
        }
        groups = [nn_params, other_params, gp_params]
        return groups
