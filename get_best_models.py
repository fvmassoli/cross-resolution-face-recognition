import os
import shutil
from tqdm import tqdm

import torch


def main():
    b = '/mnt/datone/low_resolution_face_recognition_output/model_checkpoints'
    for d in tqdm(os.listdir(b), total=len(os.listdir(b))):
        best_model_path = None
        p = os.path.join(b, d)
        best_acc = 0
        p_bar = tqdm(os.listdir(p), total=len(os.listdir(p)), desc='In folder: {}'.format(d), leave=False)
        for model in p_bar:
            dict_ = torch.load(os.path.join(p, model), map_location='cpu')
            p_bar.set_postfix(best_acc=best_acc, current_acc=dict_['best_acc'])
            if dict_['best_acc'] > best_acc:
                best_acc = dict_['best_acc']
                best_model_path = os.path.join(p, model)
        p_bar.close()
        #source = best_model_path
        #destination = '/mnt/datone/low_resolution_face_recognition_output/models_checkpoints_journal/'
        #shutil.copy(source, os.path.join(destination, best_model_path.split('/')[-2]+'--'+best_model_path.split('/')[-1]))
        print('=' * 100)
        print('=' * 100)
        print('\t Best model for: {}'
              '\n\t{}'.format(d, best_model_path))
        print('=' * 100)
        print('=' * 100)
        print(' ')
        print(' ')


if __name__ == '__main__':
    main()
