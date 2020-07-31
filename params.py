import os
import PIL


def get_params(test):
    d = dict()

    d['VGGFace2'] = 'vggface2'
    d['vggface2_train'] = 'small_test_set' if test else 'train_copied'
    d['vggface2_valid'] = 'small_test_set' if test else 'validation'
    d['vggface2_test'] = 'small_test_set' if test else 'test_mtcnn_caffe'

    d['IJBB'] = 'ijbb'
    d['ijbb_feat_extr'] = 'small_test_set' if test else 'cropped_images'

    d['IJBC'] = 'ijbc'
    d['ijbc_feat_extr'] = 'cropped_images'

    d['QMUL'] = 'qmul'
    d['qmul_imgs_base_dir'] = 'QMUL-SurvFace'

    d['TinyFace'] = 'tinyface'
    d['tinyface_feat_extr'] = 'tinyface'

    d['SCFace'] = 'scface'

    return d


def get_kwargs(data_set_name, run_mode, lowering_resolution_prob, curr_step_iterations, curriculum):
    if data_set_name == 'vggface2':
        if run_mode == 'extr_feat':
            kwargs = {
                'test': True if data_set_name == 'vggface2' else False,
                'batch_size': 48,
                'num_of_workers': 1,
                'run_mode': run_mode
            }
        else:
            kwargs = {
                'test': False,
                'algo_name': 'bilinear',
                'algo_val': PIL.Image.BILINEAR,
                'lowering_resolution_prob': lowering_resolution_prob,
                'curr_step_iterations': curr_step_iterations,
                'curriculum': curriculum,
                'valid_fix_resolution': 24,
                'batch_size': 32,
                'num_of_workers': 4,
                'run_mode': run_mode
            }
    elif data_set_name == 'ijbb':
        kwargs = {
            'algo_name': 'bilinear',
            'algo_val': PIL.Image.BILINEAR,
            'batch_size': 48,
            'num_of_workers': 1,
            'run_mode': run_mode
        }
    elif data_set_name == 'ijbc':
        kwargs = {
            'algo_name': 'bilinear',
            'algo_val': PIL.Image.BILINEAR,
            'batch_size': 48,
            'num_of_workers': 1,
            'run_mode': run_mode
        }
    elif data_set_name == 'tinyface':
        kwargs = {
            'batch_size': 48,
            'num_of_workers': 1,
            'run_mode': run_mode
        }
    elif data_set_name == 'qmul':
        if run_mode == 'train':
            kwargs = {
                'batch_size': 32,
                'num_of_workers': 4,
                'run_mode': run_mode
            }
        else:
            kwargs = {
                'batch_size': 48,
                'num_of_workers': 1,
                'run_mode': run_mode
            }
    elif data_set_name == 'scface':
        kwargs = {
            'batch_size': 32,
            'num_of_workers': 4,
            'run_mode': run_mode
        }
    else:
        kwargs = None
    return kwargs
