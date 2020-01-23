import numpy as np
import torch

from .dkl_svgp import DeepFeatureSVGP


class GPEnsemble(torch.nn.Module):
    def __init__(
            self,
            input_dim,
            target_dim,
            num_components,
            num_elites,
            gp_params
    ):
        super().__init__()
        self.component_rank = list(range(num_components))
        self.num_elites = num_elites
        components = [DeepFeatureSVGP(
            input_dim=input_dim,
            label_dim=target_dim,
            **gp_params
        ) for _ in range(num_components)]
        self.components = torch.nn.ModuleList(components)

    def random_inds(self, batch_size):
        return np.random.randint(0, self.num_elites, (batch_size,))

    def fit(self, inputs, targets, fit_args, bootstrapped=True):
        n, _ = inputs.shape
        holdout_metrics = []
        for gp in self.components:
            boot_idx = np.random.randint(0, n, (n,)) if bootstrapped else np.arange(n)
            gp_inputs, gp_targets = inputs[boot_idx], targets[boot_idx]
            metrics = gp.fit(gp_inputs, gp_targets, **fit_args)
            holdout_metrics.append((metrics['holdout_loss'], metrics['holdout_mse']))
        # rank components by holdout loss
        self.component_rank.sort(key=lambda i: holdout_metrics[i][0])
        return holdout_metrics

    def predict(self, inputs, factored=False):
        """
        inputs: np.array [n x d]
        factored: bool, if True do not aggregate predictions
        return pred_mean: np.array, pred_var: np.array
        """
        elite_idxs = self.component_rank[:self.num_elites]
        component_means, component_vars = [], []
        for i in elite_idxs:
            gp = self.components[i]
            mean, var = gp.predict(inputs)
            component_means.append(mean)
            component_vars.append(var)
        factored_means, factored_vars = np.stack(component_means), np.stack(component_vars)

        agg_mean = factored_means.mean(0)
        pred_mean_var = np.power(factored_means - agg_mean, 2).mean(0)
        pred_metrics = {
            'avg_pred_mean_var': pred_mean_var.mean()
        }
        if factored:
            pred_mean, pred_var = factored_means, factored_vars
        else:
            pred_mean = agg_mean
            pred_var = pred_mean_var + factored_vars.mean(0)

        return pred_mean, pred_var, pred_metrics
