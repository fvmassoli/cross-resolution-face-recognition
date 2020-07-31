import sys
sys.path.append('./')

import numpy as np

from vggface2_training_data_set import VGGFace2Dataset
from vggface2_test_data_set import VGGFace2TestDataset
from data_manager.core.data_manager_base_class import *


class VGGFace2(DataManagerBaseClass):
    def __init__(self, dataset_path, img_folders, transforms, device, logging, **kwargs):
        loaders_names = [
                    'VGGFace2 training', 
                    'VGGFace2 validation down sampled',
                    'VGGFace2 validation']
        self._run_mode = kwargs['run_mode']
        super(VGGFace2, self).__init__(
                                    data_set_manager_name='vggface2',
                                    loaders_names=loaders_names,
                                    device=device,
                                    **kwargs
                                )
        self._logging = logging
        self._dataset_path = dataset_path
        self._datasets = self._init_datasets(dataset_path, img_folders, transforms, **kwargs)
        self._data_loaders = self._init_data_loaders()
        self._print_summary()

    def _init_datasets(self, dataset_path, img_folders, transforms, **kwargs):
        self._logging.info(f'Initializing VGGFace2 data sets...')
        if self._run_mode == 'test':
            test_dataset = VGGFace2TestDataset(root=os.path.join(dataset_path, img_folders[2]), transforms=transforms[0])
            self._logging.info('Test datasets initialized!!!')
            return [test_dataset]
        else:
            train_dataset = VGGFace2Dataset(
                                        root=os.path.join(dataset_path, img_folders[0]),
                                        transforms=transforms[1],
                                        train=True,
                                        fix_resolution=self._fix_resolution,
                                        **kwargs
                                    )
            kwargs['lowering_resolution_prob'] = 1.0  # only for validation
            valid_dataset_lr = VGGFace2Dataset(
                                            root=os.path.join(dataset_path, img_folders[1]),
                                            transforms=transforms[0],
                                            train=False,
                                            fix_resolution=None,
                                            **kwargs
                                        )
            valid_dataset = ImageFolder(
                                    root=os.path.join(dataset_path, img_folders[1]), 
                                    transform=transforms[0]
                                )
            self._logging.info('Train datasets initialized!!!')
            return train_dataset, valid_dataset_lr, valid_dataset
            

    def _init_data_loaders(self):
        self._logging.info('Initializing VGGFace2 data loaders...')
        if self._run_mode == 'test':
            test_data_loader = DataLoader(
                                    dataset=self._datasets[0],
                                    batch_size=self._batch_size,
                                    shuffle=False,
                                    num_workers=self._num_of_workers,
                                    pin_memory=self._use_cuda
                                )
            return [test_data_loader]
        else:
            train_data_loader = DataLoader(
                                        dataset=self._datasets[0],
                                        batch_size=self._batch_size,
                                        shuffle=True,
                                        num_workers=self._num_of_workers,
                                        pin_memory=self._use_cuda
                                    )
            dataset_len = len(self._datasets[1])
            indices = list(np.arange(0, dataset_len, 30))
            split = int(np.floor(len(indices) * 0.5))
            valid_indices = indices[split:]
            tmp_valid_dataset_lr = Subset(self._datasets[1], valid_indices)
            tmp_valid_dataset = Subset(self._datasets[2], valid_indices)
            valid_data_loader_lr = DataLoader(
                                        dataset=tmp_valid_dataset_lr,
                                        batch_size=self._batch_size,
                                        num_workers=self._num_of_workers,
                                        pin_memory=self._use_cuda
                                    )
            valid_data_loader = DataLoader(
                                        dataset=tmp_valid_dataset,
                                        batch_size=self._batch_size,
                                        num_workers=self._num_of_workers,
                                        pin_memory=self._use_cuda
                                    )
            return train_data_loader, valid_data_loader_lr, valid_data_loader
            
    def _print_summary(self):
        self._logging.info("VGGFace2 dataset:")
        if self._run_mode == 'test':
            self._logging.info(
                            f'\tBatch size:           {self._batch_size}'
                            f'\n\tNumber of workers:  {self._num_of_workers}'
                            f'\n\tTest images:        {len(self._data_loaders[0].dataset)}'
                            f'\n\tTest batches:       {len(self._data_loaders[0])}'
                            f'\n\tPin Memory:         {self._use_cuda}'
                        )   
        else:
            self._logging.info(
                    f'\tBatch size:                    {self._batch_size}'
                    f'\n\tNumber of workers:           {self._num_of_workers}'
                    f'\n\tTraining images:             {len(self._data_loaders[0].dataset)}'
                    f'\n\tTraining batches:            {len(self._data_loaders[0])}'
                    f'\n\tValidation images:           {len(self._data_loaders[1].dataset)}'
                    f'\n\tValidation batches:          {len(self._data_loaders[1])}'
                    f'\n\tValidation original images:  {len(self._data_loaders[2].dataset)}'
                    f'\n\tValidation original batches: {len(self._data_loaders[2])}'
                    f'\n\tPin Memory:                  {self._use_cuda}\n'
                )

