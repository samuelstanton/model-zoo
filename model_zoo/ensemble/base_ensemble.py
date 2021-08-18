import numpy as np
import torch
import abc
import copy


class BaseEnsemble(torch.nn.Module, abc.ABC):
    """ Component-agnostic ensemble of models with NumPy interface

    Provides a standard interface through the `fit`, `predict`, `validate` and `sample` methods.
    For basic functionality, subclasses must implement an __init__ method that populates
    the `components` attribute with a ModuleList (see `model_zoo.ensemble.GPEnsemble`).
    Components should implement a `fit` and `predict` method.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

    def fit(self, dataset, fit_params, bootstrap=False):
        """
        Args:
            :param dataset (model_zoo.utils.data.Dataset)
            :param fit_params = {
                        lr=1e-3,
                        weight_decay=1e-4,
                        batch_size=32,
                        early_stopping=True,
                        wait_epochs=10,
                        wait_tol=1e-3,
                }
        """
        # allow per-component fit params
        if isinstance(fit_params, dict):
            fit_params = [fit_params] * len(self.components)
        updated_fit_params = [copy.deepcopy(self.fit_defaults)] * len(self.components)
        for default_params, passed_params in zip(updated_fit_params, fit_params):
            default_params.update(passed_params)

        val_losses = np.empty((len(self.components),))
        train_losses = np.empty_like(val_losses)
        for i, component in enumerate(self.components):
            bootstrap_id = i if bootstrap else None
            dataset.use_bootstrap(bootstrap_id)
            metrics = component.fit(dataset, updated_fit_params[i])
            val_losses[i] = metrics['val_loss'][-1]
            train_losses[i] = metrics['train_loss'][-1]
        dataset.use_bootstrap(None)  # TODO this could be implemented as a context

        # rank components by holdout loss
        self.component_rank.sort(key=lambda k: val_losses[k])
        # print(f"val loss: {val_losses}")
        # print(f"best components: {self.component_rank[:self.num_elites]}")

        val_loader = dataset.get_loader(updated_fit_params[0]['batch_size'], split='holdout')
        metrics = self.validate(val_loader)
        metrics.update(dict(
            train_loss=train_losses.mean().item(),
        ))
        return metrics

    def predict(self, inputs, *args, **kwargs):
        raise NotImplementedError

    def validate(self, val_loader, *args, **kwargs):
        """
        Args:
            inputs (np.array): [n x input_dim]
            targets (np.array): [n x target_dim]
        Returns:
            metrics (dict):
        """
        raise NotImplementedError

    def sample(self, inputs, *args, **kwargs):
        """ Draw from the true predictive distribution of the ensemble
        Args:
            inputs (np.array): [n x input_dim]
        Returns:
            samples (np.array): [n x target_dim]
        """
        raise NotImplementedError

    def reset(self, *args, **kwargs):
        raise NotImplementedError
