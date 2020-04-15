import numpy as np
from torch.utils.data import Dataset


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
            max_idxs = [seq.shape[0] - self.train_seq_len for seq in input_seqs]
            start_idxs = [np.random.randint(0, idx) for idx in max_idxs]
            stop_idxs = [idx + self.train_seq_len for idx in start_idxs]

            input_chunks = [seq[start:stop] for seq, start, stop in zip(input_seqs, start_idxs, stop_idxs)]
            target_chunks = [seq[start:stop] for seq, start, stop in zip(target_seqs, start_idxs, stop_idxs)]
            if len(input_chunks) > 1:
                return np.stack(input_chunks), np.stack(target_chunks)
            else:
                return input_chunks[0], target_chunks[0]
        else:
            raise NotImplementedError

    def sample_holdout_chunks(self):
        item = list(range(len(self.holdout_input_seqs)))
        return self[item]
