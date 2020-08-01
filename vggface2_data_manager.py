import os
import numpy as np

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset

from vggface2_custom_dataset import VGGFace2Dataset


class VGGFace2DataManager():
    def __init__(self, dataset_path, img_folders, transforms, device, logging, **kwargs):
        loaders_names = [
                    'VGGFace2 training', 
                    'VGGFace2 validation down sampled',
                    'VGGFace2 validation'
                ]
        self._dataset_path = dataset_path
        self._train_img_folders = img_folders[0]
        self._valid_img_folders = img_folders[1]
        self._train_transforms = transforms[0]
        self._valid_transforms = transforms[1]
        self._use_cuda = device == 'cuda'
        self._logging = logging
        self._kwargs = kwargs
        self._batch_size = kwargs['batch_size']
        self._num_of_workers = kwargs['num_of_workers']
        self._datasets = self._init_datasets()
        self._data_loaders = self._init_data_loaders()
        self._print_summary()

    def _init_datasets(self):
        self._logging.info(f'Initializing VGGFace2 data sets...')
        train_dataset = VGGFace2Dataset(
                                    root=os.path.join(self._dataset_path, self._train_img_folders),
                                    transforms=self._train_transforms,
                                    train=True,
                                    logging=self._logging,
                                    **self._kwargs
                                )
        valid_dataset_lr = VGGFace2Dataset(
                                        root=os.path.join(self._dataset_path, self._valid_img_folders),
                                        transforms=self._valid_transforms,
                                        train=False,
                                        logging=self._logging,
                                        **self._kwargs
                                    )
        valid_dataset = ImageFolder(
                                root=os.path.join(self._dataset_path, self._valid_img_folders), 
                                transform=self._valid_transforms
                            )
        self._logging.info('Train datasets initialized!!!')
        return train_dataset, valid_dataset_lr, valid_dataset
            
    def _init_data_loaders(self):
        self._logging.info('Initializing VGGFace2 data loaders...')
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
        self._logging.info("VGGFace2 data summary:")
        self._logging.info(
                    f'\tBatch size:                  {self._batch_size}'
                    f'\n\t\t\t\tNumber of workers:           {self._num_of_workers}'
                    f'\n\t\t\t\tTraining images:             {len(self._data_loaders[0].dataset)}'
                    f'\n\t\t\t\tTraining batches:            {len(self._data_loaders[0])}'
                    f'\n\t\t\t\tValidation images:           {len(self._data_loaders[1].dataset)}'
                    f'\n\t\t\t\tValidation batches:          {len(self._data_loaders[1])}'
                    f'\n\t\t\t\tValidation original images:  {len(self._data_loaders[2].dataset)}'
                    f'\n\t\t\t\tValidation original batches: {len(self._data_loaders[2])}'
                    f'\n\t\t\t\tPin Memory:                  {self._use_cuda}\n'
                )

    def get_datasets(self):
        return self._datasets
    
    def get_loaders(self):
        return self._data_loaders        
