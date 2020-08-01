import os
from tqdm import tqdm

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset


class DataManagerBaseClass(object):
    def __init__(self, data_set_manager_name, loaders_names, device, **kwargs):
        self._data_sets = None
        self._data_loaders = None
        self._loader_index = None
        self._run_mode = kwargs['run_mode']
        self._data_set_manager_name = data_set_manager_name
        self._loaders_names = loaders_names
        self._device = device
        self._batch_size = kwargs['batch_size']
        self._num_of_workers = kwargs['num_of_workers']
        self._progress_bars = [None for i in range(len(self._loaders_names))]
        self._use_cuda = True if self._device == 'cuda' else False

    def _init_data_sets(self, dataset_path, img_folders, transforms, **kwargs):
        raise NotImplementedError

    def _init_data_loaders(self):
        raise NotImplementedError

    def _print_summary(self):
        raise NotImplementedError

    def get_data_sets(self):
        return self._data_sets

    def get_data_loaders(self):
        return self._data_loaders

    def get_data_set_manager_name(self):
        return self._data_set_manager_name

    def get_loaders_info(self):
        self._data_loaders = [l for l in self._data_loaders if l is not None]
        return len(self._data_loaders), self._loaders_names

    def init_progress_bar_by_index(self, idx, pb_description, pb_leave=True):
        self._loader_index = idx
        self._progress_bars[self._loader_index] = tqdm(iterable=self._data_loaders[self._loader_index],
                                                       desc=pb_description,
                                                       total=len(self._data_loaders[self._loader_index]),
                                                       leave=pb_leave)
        return len(self._data_loaders[self._loader_index]), len(self._data_loaders[self._loader_index].dataset)

    def get_loader_by_index(self):
        assert self._loader_index is not None, 'self._loader_index not initialized. You need to first call ' \
                                               'the init_progress_bar_by_index() method'
        return self._data_loaders[self._loader_index]

    def close_progress_bar_by_index(self):
        self._progress_bars[self._loader_index].close()

    def update_progress_bar(self):
        self._progress_bars[self._loader_index].update(n=1)

    def get_device(self):
        return self._device
