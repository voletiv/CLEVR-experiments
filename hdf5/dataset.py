import glob
import h5py
import numpy as np
import os
import pickle

from torch.utils.data import Dataset


default_opener = lambda p_: h5py.File(p_, 'r')


class HDF5Dataset(Dataset):

    @staticmethod
    def _get_num_in_shard(shard_p, opener=default_opener):
        hdf5_root = os.path.dirname(shard_p)
        p_to_num_per_shard_p = os.path.join(hdf5_root, 'num_per_shard.pkl')
        # Speeds up filtering massively on slow file systems...
        if os.path.isfile(p_to_num_per_shard_p):
            with open(p_to_num_per_shard_p, 'rb') as f:
                p_to_num_per_shard = pickle.load(f)
                num_per_shard = p_to_num_per_shard[os.path.basename(shard_p)]
        else:
            print('\rOpening {}...'.format(shard_p), end='')
            with opener(shard_p) as f:
                num_per_shard = len(f.keys())
        return num_per_shard

    @staticmethod
    def filter_smaller_shards(file_ps, opener=default_opener):
        """
        Filter away the last shard, which is assumed to be smaller. this double checks that all other shards have the
        same number of entries.
        :param file_ps: list of .hdf5 files
        :param opener:
        :return: tuple (ps, num_per_shard) where
            ps = filtered file paths,
            num_per_shard = number of entries in all of the shards in `ps`
        """
        file_ps = sorted(file_ps)  # we assume that smallest shard is at the end
        num_per_shard_prev = None
        ps = []
        for i, p in enumerate(file_ps):
            num_per_shard = _get_num_in_shard(p, opener)
            if num_per_shard_prev is None:  # first file
                num_per_shard_prev = num_per_shard
                ps.append(p)
                continue
            if num_per_shard_prev < num_per_shard:
                raise ValueError('Expected all shards to have the same number of elements,'
                                 'except last one. Previous had {} elements, current ({}) has {}!'.format(
                                    num_per_shard_prev, p, num_per_shard))
            if num_per_shard_prev > num_per_shard:  # assuming this is the last
                is_last = i == len(file_ps) - 1
                if not is_last:
                    raise ValueError(
                            'Found shard with too few elements, and it is not the last one! {}\n'
                            'Last: {}\n'
                            'Make sure to sorte file_ps before filtering.'.format(p, file_ps[-1]))
                print('Filtering shard {}, dropping {} elements...'.format(p, num_per_shard))
                break  # is last anyways
            else:  # same numer as before, all good
                ps.append(p)
        assert num_per_shard_prev is not None
        return ps, num_per_shard_prev

    def __init__(self, clevr_hdf5_dir,
                 shuffle_shards=True,
                 opener=default_opener,
                 seed=29):
        """
        :param file_ps: list of file paths to .hdf5 files
        :param transform: transformation to apply to read HDF5 dataset. Must contain some transformation to array!
        :param shuffle_shards: if true, shards are shuffled with seed
        """
        self.clevr_hdf5_dir = clevr_hdf5_dir
        self.file_ps = glob.glob(os.path.join(clevr_hdf5_dir, '*.hdf5'))
        self.opener = opener
        self.ps, self.num_per_shard = HDF5Dataset.filter_smaller_shards(file_ps)
        self.num_of_shards = len(ps)
        if shuffle_shards:
            np.random.seed(seed)
            np.random.shuffle(self.ps)
        self.transform = transform

    def __len__(self):
        return len(self.ps) * self.num_per_shard

    def __getitem__(self, index):
        shard_idx = (index // self.num_per_shard) % self.num_of_shards
        idx_in_shard = index % self.num_per_shard
        shard_p = self.ps[shard_idx]
        with self.opener(shard_p) as f:
            image = torch.from_numpy(f['images'][str(idx_in_shard)][()][None]).float().div(255.).mul(2.).add(-1.)
            class_label = torch.tensor(f['num_of_objs'][str(idx_in_shard)][()][None])
            q_skipthought = torch.from_numpy(f['q_skipthoughts'][str(idx_in_shard)][()][None]).float()
            a_skipthought = torch.from_numpy(f['a_skipthoughts'][str(idx_in_shard)][()][None]).float()
        # Return
        return image, class_label, q_skipthought, a_skipthought
