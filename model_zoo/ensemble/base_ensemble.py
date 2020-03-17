import numpy as np
import torch

from scipy.stats import norm as Normal


class BaseEnsemble(torch.nn.Module):
    """ Component-agnostic ensemble of models with NumPy interface

    Provides a standard interface through the `fit`, `predict`, `validate` and `sample` methods.
    For basic functionality, subclasses must implement an __init__ method that populates
    the `components` attribute with a ModuleList (see `model_zoo.ensemble.GPEnsemble`).
    Components should implement a `fit` and `predict` method.
    """
    def __init__(self, input_dim, target_dim, num_components, num_elites, *args, **kwargs):
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

    def fit(self, dataset, fit_kwargs):
        """
        Args:
            dataset (model_zoo.utils.Dataset)
            fit_kwargs (dict): parameters to be passed to `component.fit`
        """
        train_inputs, train_targets = dataset.train_data
        holdout_losses = np.empty((len(self.components),))
        holdout_mses = np.empty_like(holdout_losses)
        for i, component in enumerate(self.components):
            boot_idx = dataset.bootstrap_idxs[i]
            bootstrap_data = train_inputs[boot_idx], train_targets[boot_idx]
            metrics = component.fit(bootstrap_data, dataset.holdout_data, **fit_kwargs)
            holdout_losses[i] = metrics['holdout_loss']
            holdout_mses[i] = metrics['holdout_mse']

        # rank components by holdout loss
        self.component_rank.sort(key=lambda k: holdout_losses[k])
        print(f"holdout loss: {holdout_losses}")
        print(f"holdout MSE: {holdout_mses}")
        print(f"best components: {self.component_rank[:self.num_elites]}")
        metrics = dict(
            holdout_loss=holdout_losses.mean(),
            holdout_mse=holdout_mses.mean()
        )
        return metrics

    def predict(self, inputs, factored=False):
        """
        Args:
            inputs (np.array): [n x input_dim]
            factored (bool): if True, do not aggregate predictions
        Returns:
             pred_mean (np.array)
             pred_var (np.array)
        """
        elite_idxs = self.component_rank[:self.num_elites]
        component_means, component_vars = [], []
        for i in elite_idxs:
            component = self.components[i]
            mean, var = component.predict(inputs)
            component_means.append(mean)
            component_vars.append(var)
        factored_means, factored_vars = np.stack(component_means), np.stack(component_vars)

        agg_mean = factored_means.mean(0)
        pred_mean_var = np.power(factored_means - agg_mean, 2).mean(0)
        if factored:
            pred_mean, pred_var = factored_means, factored_vars
        else:
            pred_mean = agg_mean
            pred_var = pred_mean_var + factored_vars.mean(0)

        return pred_mean, pred_var

    def validate(self, inputs, targets):
        """
        Args:
            inputs (np.array): [n x input_dim]
            targets (np.array): [n x target_dim]
        Returns:
            metrics (dict): MSE and log_prob of aggregate predictive distribution
        """
        pred_mean, pred_var = self.predict(inputs, factored=False)
        mse = ((pred_mean - targets) ** 2).mean()
        log_prob = Normal.logpdf(targets, pred_mean, np.sqrt(pred_var)).mean()
        metrics = dict(
            val_mse=mse.item(),
            val_log_prob=log_prob.item()
        )
        return metrics

    def sample(self, inputs):
        """ Draw from the true predictive distribution of the ensemble
        Args:
            inputs (np.array): [n x input_dim]
        Returns:
            samples (np.array): [n x target_dim]
        """
        n, _ = inputs.shape
        pred_mean, pred_var = self.predict(inputs, factored=True)
        element_idx = np.random.randint(0, self.num_elites, (n,))
        pred_mean = pred_mean[element_idx, np.arange(n)]
        pred_var = pred_var[element_idx, np.arange(n)]
        samples = np.random.normal(loc=pred_mean, scale=np.sqrt(pred_var))
        return samples
