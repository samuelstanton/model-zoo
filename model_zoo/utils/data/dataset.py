import numpy as np


class Dataset(object):
    """
    The Dataset object maintains a persistent holdout dataset in its state.
    """
    def __init__(self, holdout_ratio, n_bootstraps, bootstrap_size):
        self.train_inputs, self.train_targets = None, None
        self.holdout_inputs, self.holdout_targets = None, None
        self.n_train, self.n_holdout = 0, 0
        self.holdout_ratio = holdout_ratio
        self.max_n_holdout = 5000
        self.n_bootstraps = n_bootstraps
        self.bootstrap_size = bootstrap_size
        self.bootstrap_idxs = None

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

    @property
    def train_data(self):
        return self.train_inputs, self.train_targets

    @property
    def holdout_data(self):
        return self.holdout_inputs, self.holdout_targets
