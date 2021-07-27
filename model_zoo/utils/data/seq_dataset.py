import random
import numpy as np
import torch

from torch.utils.data import Dataset, WeightedRandomSampler, BatchSampler, DataLoader, TensorDataset


class SeqDataset(Dataset):
    """
    Data structure for learning from sequential data, subclasses torch.utils.data.Dataset.
    Train/holdout splits are performed at the sequence level. Random sampling of subsequences
    is implemented in the `_sample_subseqs`. Use the sampler returned by `get_weighted_sampler`
    to draw from a uniform distribution over subsequences. Use `train_seq_len=1` and `subseq_format='flat'`
    for compatibility with models that expect 2-D subsequences of length 1.
    """
    def __init__(self, train_seq_len, holdout_ratio, subseq_format='sequential',
                 n_bootstraps=1, bootstrap_size=1.):
        super().__init__()
        self.holdout_ratio = holdout_ratio
        self.train_input_seqs, self.train_target_seqs = [], []
        self.holdout_input_seqs, self.holdout_target_seqs = [], []
        if train_seq_len < 1:
            raise RuntimeError("train_seq_len must be at least 1")
        self.train_seq_len = train_seq_len
        self._subseq_format = subseq_format
        self.n_bootstraps = n_bootstraps
        self.bootstrap_size = bootstrap_size
        self.bootstrap_idxs = None
        self.current_bootstrap = None

    def __len__(self):
        inputs, _ = self.train_data
        if inputs is None:
            return 0
        num_train = inputs.shape[0]
        return num_train

    def __getitem__(self, item):
        if isinstance(item, int):
            item = [item]
        if isinstance(item, slice):
            item = list(range(item.start, item.stop, item.step))
        if isinstance(item, list):
            input_seqs = [self.train_input_seqs[i] for i in item]
            target_seqs = [self.train_target_seqs[i] for i in item]
            return self._sample_subseq(input_seqs, target_seqs)
        else:
            raise NotImplementedError

    def add_seq(self, input_seq, target_seq):
        """
        :param input_seq (np.array): [seq_len, input_dim]
        :param target_seq (np.array): [seq_len, target_dim]
        """
        # reject sequences that are too short
        seq_len, _ = input_seq.shape
        if seq_len <= self.train_seq_len:
            return None

        # sequences always start in holdout
        self.holdout_input_seqs.append(input_seq)
        self.holdout_target_seqs.append(target_seq)

        num_old = len(self)  # get number of train subseqs
        num_new = 0
        pop_inputs = pop_targets = None
        if np.random.rand() > self.holdout_ratio and len(self.holdout_input_seqs) > 1:
            # remove random sequence from holdout to add to train
            pop_idx = np.random.randint(0, len(self.holdout_input_seqs))
            pop_inputs = self.holdout_input_seqs.pop(pop_idx)
            pop_targets = self.holdout_target_seqs.pop(pop_idx)

            # count new subsequences that will be added
            # formatted_inputs, _ = format_seqs(pop_inputs, pop_targets, self.train_seq_len, self._subseq_format)
            num_new = pop_inputs.shape[0]

        # update the bootstraps
        new_bootstrap_idxs = self.get_bootstrap_idxs(0, num_new, num_new)
        if self.bootstrap_idxs is None:
            self.bootstrap_idxs = new_bootstrap_idxs
        else:
            self.bootstrap_idxs = np.concatenate(
                [self.bootstrap_idxs, num_old + new_bootstrap_idxs],
                axis=-1
            )
        # add the popped sequences to train
        if pop_inputs is not None:
            self.train_input_seqs.append(pop_inputs)
            self.train_target_seqs.append(pop_targets)

    def sample_holdout_subseqs(self):
        batch_size = min(5000, 3 * len(self.holdout_input_seqs))
        input_seqs, target_seqs = self._sample_seqs(self.holdout_input_seqs, self.holdout_target_seqs, batch_size)
        return self._sample_subseq(input_seqs, target_seqs)

    def sample_inputs(self, batch_size):
        input_seqs, target_seqs = self._sample_seqs(self.train_input_seqs, self.train_target_seqs, batch_size)
        input_subseqs, _ = self._sample_subseq(input_seqs, target_seqs)
        if self._subseq_format == 'sequential':
            return input_subseqs[:, 0]
        else:
            return input_subseqs

    def _sample_seqs(self, input_seqs, target_seqs, batch_size):
        n = len(input_seqs)
        weights = [seq.shape[0] for seq in input_seqs]
        idxs = random.choices(list(range(n)), weights=weights, k=batch_size)
        input_seqs = [input_seqs[i] for i in idxs]
        target_seqs = [target_seqs[i] for i in idxs]
        return input_seqs, target_seqs

    def _sample_subseq(self, input_seqs, target_seqs):
        max_idxs = [seq.shape[0] - self.train_seq_len for seq in input_seqs]
        start_idxs = [np.random.randint(0, idx) for idx in max_idxs]
        stop_idxs = [idx + self.train_seq_len for idx in start_idxs]

        input_subseqs = [seq[start:stop] for seq, start, stop in zip(input_seqs, start_idxs, stop_idxs)]
        target_subseqs = [seq[start:stop] for seq, start, stop in zip(target_seqs, start_idxs, stop_idxs)]

        if len(input_subseqs) > 1:
            input_subseqs = np.stack(input_subseqs)
            target_subseqs = np.stack(target_subseqs)
        else:
            input_subseqs = input_subseqs[0]
            target_subseqs = target_subseqs[0]
        if self._subseq_format == 'flat':
            input_subseqs = input_subseqs.squeeze(-2)
            target_subseqs = target_subseqs.squeeze(-2)

        return input_subseqs, target_subseqs

    def get_sampler(self, batch_size):
        weights = [seq.shape[0] for seq in self.train_input_seqs]
        w_sampler = WeightedRandomSampler(weights, num_samples=len(self))
        return BatchSampler(w_sampler, batch_size, drop_last=True)

    def get_stats(self, compat_mode='np'):
        inputs, targets = format_seqs(self.train_input_seqs, self.train_target_seqs,
                                      self.train_seq_len, self._subseq_format)
        input_dim = inputs.shape[-1]
        target_dim = targets.shape[-1]
        inputs = inputs.reshape(-1, input_dim)
        targets = targets.reshape(-1, target_dim)
        input_stats = [inputs.mean(0), inputs.std(0)]
        target_stats = [targets.mean(0), targets.std(0)]
        if compat_mode == 'torch':
            input_stats = [torch.tensor(array, dtype=torch.get_default_dtype()) for array in input_stats]
            target_stats = [torch.tensor(array, dtype=torch.get_default_dtype()) for array in target_stats]
        return input_stats, target_stats

    def get_loader(self, batch_size):
        # sampler = self.get_sampler(batch_size)
        inputs, targets = self.train_data
        if self.current_bootstrap is not None:
            idxs = self.bootstrap_idxs[self.current_bootstrap]
            inputs, targets = inputs[idxs], targets[idxs]
        dataset = TensorDataset(
            torch.tensor(inputs, dtype=torch.get_default_dtype()),
            torch.tensor(targets, dtype=torch.get_default_dtype())
        )
        return DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)

    def subseq_format(self, format):
        if format == "sequential":
            self._subseq_format = format
        elif format == "flat":
            assert self.train_seq_len == 1
            self._subseq_format = format
        else:
            raise ValueError("unrecognized format type")

    @property
    def train_data(self):
        if len(self.train_input_seqs) == 0:
            return None, None
        return format_seqs(self.train_input_seqs, self.train_target_seqs,
                           self.train_seq_len, self._subseq_format)

    @property
    def holdout_data(self):
        return format_seqs(self.holdout_input_seqs, self.holdout_target_seqs,
                           self.train_seq_len, self._subseq_format)

    def use_bootstrap(self, bootstrap_id):
        if bootstrap_id is None:
            pass
        elif isinstance(bootstrap_id, int):
            assert bootstrap_id < self.n_bootstraps
        else:
            raise ValueError('bootstrap_id should be int < self.n_bootstraps or None')
        self.current_bootstrap = bootstrap_id

    def get_bootstrap_idxs(self, start, end, n):
        n_samples = int(self.bootstrap_size * n)
        return np.random.randint(start, end, (self.n_bootstraps, n_samples))

    def reset_bootstraps(self):
        inputs, _ = format_seqs(self.train_input_seqs, self.train_target_seqs,
                                      self.train_seq_len, self._subseq_format)
        num_train = inputs.shape[0]
        self.bootstrap_idxs = self.get_bootstrap_idxs(0, num_train, num_train * self.bootstrap_size)


def format_seqs(input_seqs, target_seqs, subseq_len, subseq_format='sequential'):
    if subseq_format == 'flat':
        inputs = np.concatenate(input_seqs)
        targets = np.concatenate(target_seqs)

    elif subseq_format == 'sequential':
        seq_lens = [seq.shape[0] for seq in input_seqs]
        num_subseqs = [seq_len // subseq_len for seq_len in seq_lens]
        start_idxs = [np.random.randint(0, (seq_len % subseq_len) + 1) for seq_len in seq_lens]
        stop_idxs = [start + num_chunks * subseq_len for start, num_chunks in zip(start_idxs, num_subseqs)]
        # TODO the trimming should happen when the sequence is added
        trimmed_inputs = [seq[start:stop] for seq, start, stop in zip(input_seqs, start_idxs, stop_idxs)]
        trimmed_targets = [seq[start:stop] for seq, start, stop in zip(target_seqs, start_idxs, stop_idxs)]

        input_subseqs = [np.stack(np.split(seq, num_chunks)) for seq, num_chunks in zip(trimmed_inputs, num_subseqs)]
        target_subseqs = [np.stack(np.split(seq, num_chunks)) for seq, num_chunks in zip(trimmed_targets, num_subseqs)]
        inputs, targets = np.concatenate(input_subseqs), np.concatenate(target_subseqs)
    else:
        raise ValueError('unrecognized subsequence format')

    return inputs, targets
