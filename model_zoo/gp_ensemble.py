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

    def random_split(self, inputs, targets, n_holdout):
        n, _ = inputs.shape
        shuffle_idx = np.arange(n)
        np.random.shuffle(shuffle_idx)
        inputs, targets = inputs[shuffle_idx], targets[shuffle_idx]
        train_data = inputs[n_holdout:], targets[n_holdout:]
        holdout_data = inputs[:n_holdout], targets[:n_holdout]
        return train_data, holdout_data

    def fit(self, dataset, fit_args):
        train_inputs, train_targets = dataset.train_data
        holdout_losses = np.empty((len(self.components),))
        holdout_mses = np.empty_like(holdout_losses)
        for i, gp in enumerate(self.components):
            boot_idx = dataset.bootstrap_idxs[i]
            bootstrap_data = train_inputs[boot_idx], train_targets[boot_idx]
            metrics = gp.fit(bootstrap_data, dataset.holdout_data, **fit_args)
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
        if factored:
            pred_mean, pred_var = factored_means, factored_vars
        else:
            pred_mean = agg_mean
            pred_var = pred_mean_var + factored_vars.mean(0)

        return pred_mean, pred_var
