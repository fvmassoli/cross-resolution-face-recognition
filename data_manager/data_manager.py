import os
import PIL

import torchvision.transforms as t

from params import get_params, get_kwargs
from .ijbb.ijbb_data_manager import IJBB
from .ijbc.ijbc_data_manager import IJBC
from .scface.scface_data_manager import SCface
from .qmul.qmul_data_manager import QMULSurvFace
from .vggface2.vggface2_data_manager import VGGFace2
from .tinyface.tinyface_data_manager import TinyFace


class DataManager(object):
    def __init__(self,
                 test,
                 data_set_name=None,
                 super_resolved_images=False,
                 device='cpu',
                 run_mode=None,
                 lowering_resolution_prob=0.0,
                 fix_resolution=False,
                 curriculum=False,
                 curr_step_iterations=10):
        self._params = get_params(test)
        self._data_set_name = data_set_name
        self._super_resolved_images = super_resolved_images
        self._data_set_path = os.path.join(self._params['DATA_SET_MAIN_PATH'], self._data_set_name.split('-')[0])
        self._device = device
        self._run_mode = run_mode
        self._lowering_resolution_prob = lowering_resolution_prob
        self._fix_resolution = fix_resolution
        self._curriculum = curriculum
        self._curr_step_iterations = curr_step_iterations
        self._check_file_existence()
        self._data_set_manager = self._load_specific_data_manager()

    def _check_file_existence(self):
        _p = os.path.join(self._data_set_path)
        assert os.path.exists(_p.split('-')[0]), "Data set not found at: {}".format(_p.split('-')[0])

    def _load_specific_data_manager(self):

        kwargs = get_kwargs(self._data_set_name, self._run_mode, self._lowering_resolution_prob,
                            self._curr_step_iterations, self._curriculum)
        transf = [self._get_transforms(mode='eval')]

        ###########################
        ## Load VGGFace2 dataset ##
        ###########################
        if self._data_set_name == self._params['VGGFace2'] or self._data_set_name == self._params['VGGFace2-500']:

            if not self._run_mode == 'extr_feat':
                transf.append(self._get_transforms(mode='train'))

            if self._data_set_name == self._params['VGGFace2']:
                img_folders = [self._params['vggface2_train'],
                               self._params['vggface2_valid'],
                               self._params['vggface2_test']]
            else:
                img_folders = [self._params['vggface2_500_train'],
                               self._params['vggface2_500_validation'],
                               self._params['vggface2_500_test']]

            return VGGFace2(data_set_path=self._data_set_path,
                            data_set_name=self._data_set_name,
                            img_folders=img_folders,
                            fix_resolution=self._fix_resolution,
                            transforms=transf,
                            device=self._device,
                            **kwargs)

        ###########################
        ### Load IJB-B dataset ####
        ###########################
        elif self._data_set_name == self._params['IJBB']:
            return IJBB(data_set_path=self._data_set_path,
                        img_folders=self._params['ijbb_feat_extr'],
                        transforms=transf,
                        device=self._device,
                        **kwargs)

        ###########################
        ### Load IJB-C dataset ####
        ###########################
        elif self._data_set_name == self._params['IJBC']:
            return IJBC(data_set_path=self._data_set_path,
                        img_folders=self._params['ijbc_feat_extr'],
                        transforms=transf,
                        device=self._device,
                        **kwargs)

        ###########################
        ## Load TinyFace dataset ##
        ###########################
        elif self._data_set_name == self._params['TinyFace']:
            return TinyFace(data_set_path=self._data_set_path,
                            super_resolved_images=self._super_resolved_images,
                            transforms=transf,
                            device=self._device,
                            **kwargs)

        ###########################
        ## Load QMUL-SF  dataset ##
        ###########################
        elif self._data_set_name == self._params['QMUL']:
            return QMULSurvFace(data_set_path=self._data_set_path,
                                super_resolved_images=self._super_resolved_images,
                                img_folders=self._params['qmul_imgs_base_dir'],
                                transforms=transf,
                                device=self._device,
                                **kwargs)

        ###########################
        ## Load SCface   dataset ##
        ###########################
        elif self._data_set_name == self._params['SCface']:
            return SCface(data_set_path=self._data_set_path,
                          super_resolved_images=self._super_resolved_images,
                          img_folders=self._params['scface_imgs_base_dir'],
                          transforms=transf,
                          device=self._device,
                          **kwargs)

    @staticmethod
    def _subtract_mean(x):
        mean_vector = [91.4953, 103.8827, 131.0912]
        x *= 255.
        x[0] -= mean_vector[0]
        x[1] -= mean_vector[1]
        x[2] -= mean_vector[2]
        return x

    def _get_transforms(self, resize=256, grayed_prob=0.2, crop_size=224):
        def subtract_mean(x):
            mean_vector = [91.4953, 103.8827, 131.0912]
            x *= 255.
            x[0] -= mean_vector[0]
            x[1] -= mean_vector[1]
            x[2] -= mean_vector[2]
            return x
        if mode=='train':
            return t.Compose([
                        t.Resize(resize),
                        t.RandomGrayscale(p=grayed_prob),
                        t.RandomCrop(crop_size),
                        t.ToTensor(),
                        t.Lambda(lambda x: self._subtract_mean(x))
                    ])
        else:
            return t.Compose([
                        t.Resize(resize),
                        t.CenterCrop(crop_size),
                        t.ToTensor(),
                        t.Lambda(lambda x: self._subtract_mean(x))
                    ])

    def get_dataset_manager(self):
        return self._data_set_manager
