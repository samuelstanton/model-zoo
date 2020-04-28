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
    def __init__(self, train_seq_len, holdout_ratio, subseq_format='sequential'):
        super().__init__()
        self.holdout_ratio = holdout_ratio
        self.train_input_seqs, self.train_target_seqs = [], []
        self.holdout_input_seqs, self.holdout_target_seqs = [], []
        if train_seq_len < 1:
            raise RuntimeError("train_seq_len must be at least 1")
        self.train_seq_len = train_seq_len
        self._subseq_format = subseq_format

    def __len__(self):
        return len(self.train_input_seqs)

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
        seq_len, _ = input_seq.shape
        if seq_len <= self.train_seq_len:
            pass  # reject sequences that are too short
        else:
            self.holdout_input_seqs.append(input_seq)
            self.holdout_target_seqs.append(target_seq)
            if np.random.rand() > self.holdout_ratio:
                pop_idx = np.random.randint(0, len(self.holdout_input_seqs))
                self.train_input_seqs.append(self.holdout_input_seqs.pop(pop_idx))
                self.train_target_seqs.append(self.holdout_target_seqs.pop(pop_idx))

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

    def get_loader(self, batch_size):
        # sampler = self.get_sampler(batch_size)
        inputs, targets = format_seqs(self.train_input_seqs, self.train_target_seqs,
                                      self.train_seq_len, self._subseq_format)
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

    def get_holdout_data(self):
        return format_seqs(self.holdout_input_seqs, self.holdout_target_seqs,
                           self.train_seq_len, self._subseq_format)


def format_seqs(input_seqs, target_seqs, subseq_len, subseq_format='sequential'):
    seq_lens = [seq.shape[0] for seq in input_seqs]
    num_subseqs = [seq_len // subseq_len for seq_len in seq_lens]
    start_idxs = [np.random.randint(0, (seq_len % subseq_len) + 1) for seq_len in seq_lens]
    stop_idxs = [start + num_chunks * subseq_len for start, num_chunks in zip(start_idxs, num_subseqs)]
    trimmed_inputs = [seq[start:stop] for seq, start, stop in zip(input_seqs, start_idxs, stop_idxs)]
    trimmed_targets = [seq[start:stop] for seq, start, stop in zip(target_seqs, start_idxs, stop_idxs)]

    input_subseqs = [np.stack(np.split(seq, num_chunks)) for seq, num_chunks in zip(trimmed_inputs, num_subseqs)]
    target_subseqs = [np.stack(np.split(seq, num_chunks)) for seq, num_chunks in zip(trimmed_targets, num_subseqs)]

    res = [np.concatenate(input_subseqs), np.concatenate(target_subseqs)]
    if subseq_format == 'flat':
        return [array.squeeze(1) for array in res]
    else:
        return res
