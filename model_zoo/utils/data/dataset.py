import numpy as np
import torch

from torch.utils.data import DataLoader

from model_zoo.utils import streaming


class Dataset(object):
    """
    The Dataset object maintains a persistent holdout dataset in its state.
    """
    def __init__(self, holdout_ratio, n_bootstraps, bootstrap_size, max_num_holdout=5000):
        self.train_inputs, self.train_targets = None, None
        self.holdout_inputs, self.holdout_targets = None, None
        self.n_train, self.n_holdout = 0, 0
        self.holdout_ratio = holdout_ratio
        self.max_n_holdout = max_num_holdout
        self.n_bootstraps = n_bootstraps
        self.bootstrap_size = bootstrap_size
        self.bootstrap_idxs = None
        self.current_bootstrap = None
        self.input_welford = streaming.Welford()
        self.target_welford = streaming.Welford()

    def __len__(self):
        return self.n_train

    def __getitem__(self, item):
        return self.train_inputs[item], self.train_targets[item]

    def get_bootstrap_idxs(self, start, end, n):
        n_samples = int(self.bootstrap_size * n)
        return np.random.randint(start, end, (self.n_bootstraps, n_samples))

    def reset_bootstraps(self):
        self.bootstrap_idxs = self.get_bootstrap_idxs(0, self.n_train, self.n_train)

    def add_new_data(self, inputs, targets):
        """
        Add new observations to the dataset. New training data will be randomly selected from
        both the new data and the existing holdout set.

        Args:
            inputs (np.array): [n_new x d]
            targets (np.array): [n_new x t]

        """
        n_new, _ = inputs.shape
        n_new_holdout = int(self.holdout_ratio * n_new)
        n_new_train = n_new - n_new_holdout
        self.n_holdout += n_new_holdout
        self.n_holdout = min(self.n_holdout, self.max_n_holdout)
        new_bootstrap_idxs = self.get_bootstrap_idxs(0, n_new_train, n_new_train)

        if self.train_inputs is None:
            new_train, new_holdout = self._random_split(inputs, targets)
            self.train_inputs, self.train_targets = new_train
            self.bootstrap_idxs = new_bootstrap_idxs

        else:
            new_train, new_holdout = self._random_split(
                np.concatenate([self.holdout_inputs, inputs]),
                np.concatenate([self.holdout_targets, targets]),
            )
            self.train_inputs, self.train_targets = (
                np.concatenate([self.train_inputs, new_train[0]]),
                np.concatenate([self.train_targets, new_train[1]])
            )
            self.bootstrap_idxs = np.concatenate(
                [self.bootstrap_idxs, self.n_train + new_bootstrap_idxs],
                axis=-1
            )

        [self.input_welford.add_data(x) for x in new_train[0]]
        [self.target_welford.add_data(y) for y in new_train[1]]
        self.n_train, _ = self.train_inputs.shape
        self.holdout_inputs, self.holdout_targets = new_holdout

    def _random_split(self, inputs, targets):
        n, _ = inputs.shape
        shuffle_idx = np.arange(n)
        np.random.shuffle(shuffle_idx)
        inputs, targets = inputs[shuffle_idx], targets[shuffle_idx]
        train_data = inputs[self.n_holdout:], targets[self.n_holdout:]
        holdout_data = inputs[:self.n_holdout], targets[:self.n_holdout]
        return train_data, holdout_data

    def get_loader(self, batch_size, split='train', drop_last=False):
        if split == 'train':
            inputs, targets = self.train_data
            if self.current_bootstrap is not None:
                idxs = self.bootstrap_idxs[self.current_bootstrap]
                inputs, targets = inputs[idxs], targets[idxs]
        elif split == 'holdout':
            inputs, targets = self.holdout_data
        else:
            raise ValueError('unrecognized split')
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(inputs, dtype=torch.get_default_dtype(), device=torch.device('cpu')),
            torch.tensor(targets, dtype=torch.get_default_dtype(), device=torch.device('cpu'))
        )
        return DataLoader(dataset, shuffle=True, batch_size=batch_size, drop_last=drop_last)

    @property
    def holdout_data(self):
        return self.holdout_inputs, self.holdout_targets

    @property
    def train_data(self):
        return self.train_inputs, self.train_targets

    def use_bootstrap(self, bootstrap_id):
        if bootstrap_id is None:
            pass
        elif isinstance(bootstrap_id, int):
            assert bootstrap_id < self.n_bootstraps
        else:
            raise ValueError('bootstrap_id should be int or None')
        self.current_bootstrap = bootstrap_id

    def get_stats(self, compat_mode='np'):
        input_stats = (
            self.input_welford.mean(),
            np.clip(self.input_welford.std(), 1e-6, None)
        )
        target_stats = (
            self.target_welford.mean(),
            np.clip(self.target_welford.std(), 1e-6, None)
        )
        if compat_mode == 'torch':
            input_stats = [torch.tensor(array, dtype=torch.get_default_dtype()) for array in input_stats]
            target_stats = [torch.tensor(array, dtype=torch.get_default_dtype()) for array in target_stats]
        return input_stats, target_stats
