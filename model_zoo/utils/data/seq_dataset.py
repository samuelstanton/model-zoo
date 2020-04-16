import random
import numpy as np

from torch.utils.data import Dataset, WeightedRandomSampler, BatchSampler


class SeqDataset(Dataset):
    def __init__(self, train_seq_len, holdout_ratio):
        super().__init__()
        self.holdout_ratio = holdout_ratio
        self.train_input_seqs, self.train_target_seqs = [], []
        self.holdout_input_seqs, self.holdout_target_seqs = [], []
        self.train_seq_len = train_seq_len

    def add_seq(self, input_seq, target_seq):
        """
        :param input_seq (np.array): [seq_len, input_dim]
        :param target_seq (np.array): [seq_len, target_dim]
        :return:
        """
        seq_len, _ = input_seq.shape
        if seq_len < self.train_seq_len:
            pass

        self.holdout_input_seqs.append(input_seq)
        self.holdout_target_seqs.append(target_seq)
        if np.random.rand() < self.holdout_ratio:
            pass
        else:
            self.train_input_seqs.append(self.holdout_input_seqs.pop(0))
            self.train_target_seqs.append(self.holdout_target_seqs.pop(0))

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
            return self._sample_chunks(input_seqs, target_seqs)
        else:
            raise NotImplementedError

    def get_weighted_sampler(self, batch_size):
        weights = [seq.shape[0] for seq in self.train_input_seqs]
        w_sampler = WeightedRandomSampler(weights, num_samples=len(self))
        return BatchSampler(w_sampler, batch_size, drop_last=True)

    def sample_holdout_chunks(self):
        batch_size = min(5000, 3 * len(self.holdout_input_seqs))
        input_seqs, target_seqs = self._sample_seqs(self.holdout_input_seqs, self.holdout_target_seqs, batch_size)
        return self._sample_chunks(input_seqs, target_seqs)

    def sample_inputs(self, batch_size):
        input_seqs, target_seqs = self._sample_seqs(self.train_input_seqs, self.train_target_seqs, batch_size)
        input_chunks, _ = self._sample_chunks(input_seqs, target_seqs)
        return input_chunks[:, 0]

    def _sample_seqs(self, input_seqs, target_seqs, batch_size):
        n = len(input_seqs)
        weights = [seq.shape[0] for seq in input_seqs]
        idxs = random.choices(list(range(n)), weights=weights, k=batch_size)
        input_seqs = [input_seqs[i] for i in idxs]
        target_seqs = [target_seqs[i] for i in idxs]
        return input_seqs, target_seqs

    def _sample_chunks(self, input_seqs, target_seqs):
        max_idxs = [seq.shape[0] - self.train_seq_len for seq in input_seqs]
        start_idxs = [np.random.randint(0, idx) for idx in max_idxs]
        stop_idxs = [idx + self.train_seq_len for idx in start_idxs]

        input_chunks = [seq[start:stop] for seq, start, stop in zip(input_seqs, start_idxs, stop_idxs)]
        target_chunks = [seq[start:stop] for seq, start, stop in zip(target_seqs, start_idxs, stop_idxs)]
        if len(input_chunks) > 1:
            return np.stack(input_chunks), np.stack(target_chunks)
        else:
            return input_chunks[0], target_chunks[0]
