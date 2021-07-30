import numpy as np
import math
import torch

from torch.nn import ModuleList
import torch.nn.functional as F

from model_zoo.ensemble import BaseEnsemble
from model_zoo.classifier import MaxLikelihoodClassifier

from model_zoo import utils


class MaxLikelihoodClassifierEnsemble(BaseEnsemble):
    """ Ensemble of fully-connected neural net regression models
    """
    def __init__(self, input_dim, num_classes, num_components,
                 num_elites, model_class, model_kwargs, fit_params):
        """
        Args:
            input_dim (int)
            target_dim (int)
            num_components (int)
            num_elites (int)
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_components = num_components
        self.component_rank = list(range(num_components))
        self.num_elites = num_elites
        self.components = None
        self.fit_defaults = fit_params
        components = [
            MaxLikelihoodClassifier(
                input_dim,
                num_classes,
                model_class,
                model_kwargs,
            ) for _ in range(num_components)
        ]
        self.components = ModuleList(components)

    def predict(self, inputs, factored=False, compat_mode='np'):
        """
        Args:
            inputs (np.array): [n x input_dim]
            factored (bool): if True, do not aggregate predictions
        Returns:
             logits (np.array)
        """
        if self.num_elites < self.num_components:
            elite_idxs = self.component_rank[:self.num_elites]
        else:
            elite_idxs = np.arange(self.num_elites)

        component_logits = []
        for i in elite_idxs:
            component = self.components[i]
            logits = component.predict(inputs, compat_mode=compat_mode)
            component_logits.append(logits)

        if compat_mode == 'np':
            factored_logits = np.stack(component_logits)
        elif compat_mode == 'torch':
            factored_logits = torch.stack(component_logits)
        else:
            raise ValueError("unrecognized compatibility mode, use 'np' (NumPy) or 'torch' (PyTorch)")

        if factored:
            return factored_logits
        else:
            factored_logp = F.log_softmax(factored_logits, dim=-1)
            return torch.logsumexp(factored_logp, dim=0) - math.log(factored_logits.shape[0])

    def validate(self, val_loader, *args, **kwargs):
        """
        Args:
            inputs (np.array): [n x input_dim]
            targets (np.array): [n x target_dim]
        Returns:
            metrics (dict)
        """
        metrics = dict(val_acc=0., val_nll=0.)
        for inputs, targets in val_loader:
            self.reset()
            with torch.no_grad():
                avg_logits = self.predict(inputs, factored=False, compat_mode='torch')
            targets = targets.to(avg_logits.device).long()

            metrics['val_acc'] += utils.metrics.top_k_accuracy(avg_logits, targets, k=1) / len(val_loader)
            metrics['val_nll'] += F.log_softmax(avg_logits, dim=-1)[..., targets].mean().item() / len(val_loader)

        return metrics

    def sample(self, inputs, *args, **kwargs):
        """ Draw from the true predictive distribution of the ensemble
        Args:
            inputs (np.array): [n x input_dim]
        Returns:
            samples (np.array): [n x target_dim]
        """
        raise NotImplementedError

    def reset(self):
        for component in self.components:
            component.reset()
