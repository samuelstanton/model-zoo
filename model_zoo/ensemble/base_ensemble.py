import numpy as np
import torch
import copy

from scipy.stats import norm as Normal

from model_zoo.utils.metrics import quantile_calibration


class BaseEnsemble(torch.nn.Module):
    """ Component-agnostic ensemble of models with NumPy interface

    Provides a standard interface through the `fit`, `predict`, `validate` and `sample` methods.
    For basic functionality, subclasses must implement an __init__ method that populates
    the `components` attribute with a ModuleList (see `model_zoo.ensemble.GPEnsemble`).
    Components should implement a `fit` and `predict` method.
    """
    def __init__(self, input_dim, target_dim, num_components, num_elites, fit_params, *args, **kwargs):
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

    def fit(self, dataset, fit_params, bootstrap=False):
        """
        Args:
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
        """
        updated_fit_params = copy.deepcopy(self.fit_defaults)
        updated_fit_params.update(fit_params)

        holdout_losses = np.empty((len(self.components),))
        train_losses = np.empty_like(holdout_losses)
        holdout_mses = np.empty_like(holdout_losses)
        for i, component in enumerate(self.components):
            bootstrap_id = i if bootstrap else None
            dataset.use_bootstrap(bootstrap_id)
            metrics = component.fit(dataset, updated_fit_params)
            holdout_losses[i] = metrics['holdout_loss'][-1]
            train_losses[i] = metrics['train_loss'][-1]
            holdout_mses[i] = metrics['holdout_mse']
        dataset.use_bootstrap(None)  # TODO this could be implemented as a context

        # rank components by holdout loss
        self.component_rank.sort(key=lambda k: holdout_losses[k])
        print(f"holdout loss: {holdout_losses}")
        print(f"holdout MSE: {holdout_mses}")
        print(f"best components: {self.component_rank[:self.num_elites]}")

        holdout_metrics = self.validate(*dataset.get_holdout_data())
        metrics = dict(
            train_loss=train_losses.mean().item(),
            holdout_nll=holdout_metrics['val_nll'],
            holdout_mse=holdout_metrics['val_mse'],
            holdout_ece=holdout_metrics['ece']
        )
        return metrics

    def predict(self, inputs, factored=False, compat_mode='np'):
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

    def validate(self, inputs, targets):
        """
        Args:
            inputs (np.array): [n x input_dim]
            targets (np.array): [n x target_dim]
        Returns:
            metrics (dict): MSE and log_prob of aggregate predictive distribution
        """
        self.reset()

        agg_mean, agg_var = self.predict(inputs, factored=False)
        metrics = quantile_calibration(torch.tensor(agg_mean), torch.tensor(agg_var).sqrt(),
                                       torch.tensor(targets))

        pred_means, pred_vars = self.predict(inputs, factored=True)
        mse = ((pred_means.mean(0) - targets) ** 2).mean()
        targets = np.tile(targets, (self.num_elites, 1, 1))
        nll = -Normal.logpdf(targets, pred_means, np.sqrt(pred_vars)).mean()
        metrics.update(dict(
            val_mse=mse.item(),
            val_nll=nll.item()
        ))
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

    def reset(self):
        for component in self.components:
            component.reset()
