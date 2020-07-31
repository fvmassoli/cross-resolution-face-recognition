import sys
sys.path.append('./')

import os
import PIL

from utils import get_transforms
from ijbb.ijbb_data_manager import IJBB
from ijbc.ijbc_data_manager import IJBC
from scface.scface_data_manager import SCface
from qmul.qmul_data_manager import QMULSurvFace
from vggface2.vggface2_data_manager import VGGFace2
from tinyface.tinyface_data_manager import TinyFace


class DataManager(object):
    def __init__(self, dataset_name, super_resolved_images, device, run_mode,
                lowering_resolution_prob, curriculum, curr_step_iterations):
        self._dataset_name = dataset_name
        self._super_resolved_images = super_resolved_images
        self._device = device
        self._run_mode = run_mode
        self._lowering_resolution_prob = lowering_resolution_prob
        self._curriculum = curriculum
        self._curr_step_iterations = curr_step_iterations    
        self._data_set_manager = self._load_specific_data_manager()

    def _load_specific_data_manager(self):
        kwargs = {
            'run_mode': self._run_mode, 
            'lowering_resolution_prob': self._lowering_resolution_prob,
            'curr_step_iterations': self._curr_step_iterations, 
            'curriculum': self._curriculum
        }
        transf = [get_transforms(mode='eval')]
        ###########################
        ## Load VGGFace2 dataset ##
        ###########################
        if self._dataset_name == 'vggface2':
            if not self._run_mode == 'extr_feat':
                transf.append(get_transforms(mode='train'))
            return VGGFace2(data_set_path=self._data_set_path,
                            data_set_name=self._data_set_name,
                            img_folders=['train_copied', 'validation'],
                            fix_resolution=self._fix_resolution,
                            transforms=transf,
                            device=self._device,
                            **kwargs)
        ###########################
        ### Load IJB-B dataset ####
        ###########################
        elif self._dataset_name == 'ijbb':
            return IJBB(data_set_path=self._data_set_path,
                        img_folders='cropped_images',
                        transforms=transf,
                        device=self._device,
                        **kwargs)
        ###########################
        ### Load IJB-C dataset ####
        ###########################
        elif self._dataset_name == 'ijbc':
            return IJBC(data_set_path=self._data_set_path,
                        img_folders='cropped_images',
                        transforms=transf,
                        device=self._device,
                        **kwargs)
        ###########################
        ## Load TinyFace dataset ##
        ###########################
        elif self._dataset_name == 'tinyface':
            return TinyFace(data_set_path=self._data_set_path,
                            super_resolved_images=self._super_resolved_images,
                            transforms=transf,
                            device=self._device,
                            **kwargs)
        ###########################
        ## Load QMUL-SF  dataset ##
        ###########################
        elif self._dataset_name == 'qmul':
            return QMULSurvFace(data_set_path=self._data_set_path,
                                super_resolved_images=self._super_resolved_images,
                                img_folders='QMUL-SurvFace',
                                transforms=transf,
                                device=self._device,
                                **kwargs)
        ###########################
        ## Load SCface   dataset ##
        ###########################
        elif self._dataset_name == 'scface':
            return SCface(data_set_path=self._data_set_path,
                          super_resolved_images=self._super_resolved_images,
                          img_folders='scface',
                          transforms=transf,
                          device=self._device,
                          **kwargs)

    def get_dataset_manager(self):
        return self._data_set_manager
