import torch
import numpy as np

from torch.nn import ModuleList

from model_zoo.ensemble import BaseEnsemble
from model_zoo.regression import MaxLikelihoodRegression

from model_zoo.utils.metrics import quantile_calibration


class MaxLikelihoodRegEnsemble(BaseEnsemble):
    """ Ensemble of fully-connected neural net regression models
    """
    def __init__(self, input_dim, target_dim, num_components,
                 num_elites, model_class, model_kwargs, fit_params, mode='prob'):
        """
        Args:
            input_dim (int)
            target_dim (int)
            num_components (int)
            num_elites (int)
        """
        super().__init__()
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.num_components = num_components
        self.component_rank = list(range(num_components))
        self.num_elites = num_elites
        self.components = None
        self.fit_defaults = fit_params
        components = [
            MaxLikelihoodRegression(
                input_dim,
                target_dim,
                model_class,
                model_kwargs,
                mode
            ) for _ in range(num_components)
        ]
        self.components = ModuleList(components)

    def predict(self, inputs, factored=False, compat_mode='np'):
        """
        Args:
            inputs (np.array): [n x input_dim]
            factored (bool): if True, do not aggregate predictions
        Returns:
             pred_mean (np.array)
             pred_var (np.array)
        """
        if self.num_elites < self.num_components:
            elite_idxs = self.component_rank[:self.num_elites]
        else:
            elite_idxs = np.arange(self.num_elites)

        component_means, component_vars = [], []
        for i in elite_idxs:
            component = self.components[i]
            mean, var = component.predict(inputs, compat_mode=compat_mode)
            component_means.append(mean)
            component_vars.append(var)

        if compat_mode == 'np':
            factored_means, factored_vars = np.stack(component_means), np.stack(component_vars)
            agg_mean = factored_means.mean(0)
            pred_mean_var = np.power(factored_means - agg_mean, 2).mean(0)
        elif compat_mode == 'torch':
            factored_means, factored_vars = torch.stack(component_means), torch.stack(component_vars)
            agg_mean = factored_means.mean(0)
            pred_mean_var = torch.pow(factored_means - agg_mean, 2).mean(0)
        else:
            raise ValueError("unrecognized compatibility mode, use 'np' (NumPy) or 'torch' (PyTorch)")

        if factored:
            pred_mean, pred_var = factored_means, factored_vars
        else:
            pred_mean = agg_mean
            pred_var = pred_mean_var + factored_vars.mean(0)

        return pred_mean, pred_var

    def validate(self, val_loader):
        """
        Args:
            inputs (np.array): [n x input_dim]
            targets (np.array): [n x target_dim]
        Returns:
            metrics (dict): MSE and log_prob of aggregate predictive distribution
        """
        metrics = {}
        for inputs, targets in val_loader:
            self.reset()
            with torch.no_grad():
                agg_mean, agg_var = self.predict(inputs, factored=False, compat_mode='torch')
                pred_means, pred_vars = self.predict(inputs, factored=True, compat_mode='torch')
            targets = targets.to(agg_mean)

            batch_metrics = quantile_calibration(agg_mean, agg_var.sqrt(), targets)
            batch_metrics['val_mse'] = ((pred_means.mean(0) - targets) ** 2).mean().item()

            targets = targets.expand(self.num_elites, -1, -1)
            pred_dist = torch.distributions.Normal(pred_means, pred_vars.sqrt())
            batch_metrics['val_nll'] = -pred_dist.log_prob(targets).mean().item()

            for key in batch_metrics.keys():
                metrics.setdefault(key, 0.)
                metrics[key] += batch_metrics[key] / len(val_loader)

        return metrics

    def sample(self, inputs, compat_mode='np'):
        """ Draw from the true predictive distribution of the ensemble
        Args:
            inputs (np.array): [n x input_dim]
        Returns:
            samples (np.array): [n x target_dim]
        """
        n, _ = inputs.shape
        pred_mean, pred_var = self.predict(inputs, factored=True, compat_mode=compat_mode)
        element_idx = np.random.randint(0, self.num_elites, (n,))
        pred_mean = pred_mean[element_idx, np.arange(n)]
        pred_var = pred_var[element_idx, np.arange(n)]
        if compat_mode == 'np':
            samples = np.random.normal(loc=pred_mean, scale=np.sqrt(pred_var))
        else:
            dist = torch.distributions.Normal(pred_mean, pred_var.sqrt())
            samples = dist.rsample()

        return samples

    def reset(self):
        for component in self.components:
            component.reset()
