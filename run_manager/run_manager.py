import numpy as np
from tqdm import tqdm

import torch.nn.functional as F

from .utils import *
from params import get_params


class RunManager(object):
    def __init__(self, student, teacher, optimizer, scheduler, data_manager, dataset, lowering_resolution_prob,
                device, curriculum, epochs, lambda_, train_steps, run_mode, super_resolved_images, out_dir):
        self._studet = student
        self._teacher = teacher
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._data_manager = data_manager
        self._dataset = dataset
        self._lowering_resolution_prob = lowering_resolution_prob
        self._device = device
        self._curriculum = curriculum
        self._epochs = epochs
        self._lambda = lambda_
        self._train_steps = train_steps
        self._run_mode = run_mode
        self._super_resolved_images = super_resolved_images
        self._out_dir = out_dir
        self._features_type = np.float32

    def _eval_batch(self, loader_idx, items):
        curr_index = -1
        lowering_resolution_prob = -1
        items = [item for item in items]

        if self._dataset == 'vggface2':
            if loader_idx == 2:  # ImageFolder for original sized images
                batch = items[0]
                batch_original = items[0]
                labels = items[1]
            else:  # custom data set for down sampled images
                batch = items[0]
                batch_original = items[1]
                labels = items[2]
                curr_index = items[3]
                lowering_resolution_prob = items[4]

            teacher_features, teacher_logits = self._teacher(batch_original)

        else:
            batch = items[0]
            labels = items[1]
            teacher_features, teacher_logits = None, None

        student_features, student_logits = self._student(batch)

        y_hat = student_logits.argmax(dim=1)
        correct_lr = (y_hat == labels).sum().item()

        loss = F.cross_entropy(student_logits, labels) + self._lambda*F.mse_loss(student_features, teacher_features)


        return loss, student_logits, labels, correct_lr, curr_index, lowering_resolution_prob

















    def run(self):
        if self._run_mode == 'train' and (self._dataset == 'vggface2' or self._dataset == 'vggface2-500'):
            self._train_stats_csv, self._valid_stats_csv, self._valid_stats_orig_csv = create_statistics_csv_file(self._time,
            self._valid_test(0)

            if self._dataset == 'vggface2':
                [self._train(epoch) for epoch in range(self._epochs)]
            elif self._dataset == 'vggface2-500':
                [self._train(epoch) for epoch in range(self._epochs)]
                self._valid_test(0, test_model=True)
            else:
                print(f'Training procedure not implemented for dataset: {self._dataset}')
        else:
            self._extract_features()



















    

    
    def _train(self, epoch, kwargs):

        self._model_manager.set_train_mode()

        batch_accumulation = 8

        best_acc = 0
        loader_idx = 0

        i = 60
        print('='*i)
        print('=' * i)
        print('\n\tTraining at epoch: {}'
              '\n\t\tLearning rate:   {}\n'.format(epoch+1, get_current_learning_rate(self._optimizer)))

        loader_len, dataset_len = self._data_manager.init_progress_bar_by_index(idx=loader_idx,
                                                                                pb_description='Model Training',
                                                                                pb_leave=False)
        local_data_loader = self._data_manager.get_loader_by_index()

        j = 1
        l_ = 0.0
        correct_ = 0.0
        n_samples_ = 0
        nb_backward_steps = 0
        self._optimizer.zero_grad()

        for batch_idx, items in enumerate(local_data_loader):

            if nb_backward_steps == self._steps_before_validation_run:
                nb_backward_steps = 0
                v_l_, tmp_best_acc = self._valid_test(epoch)
                if self._use_learning_rate_scheduler:
                    self._scheduler.step(v_l_, epoch+1)
                if tmp_best_acc > best_acc:
                    best_acc = tmp_best_acc
                    save_model_checkpoint(batch_idx + 1, best_acc, self._time,
                                          self._model_manager.get_model_state_dict(), epoch+1,
                                          self._outputs_folders[1], **kwargs)
                self._model_manager.set_train_mode()

            loss, logits, labels, tmp_correct_, curr_index, lowering_resolution_prob = self._eval_batch(loader_idx, items)

            l_ += loss.item()
            correct_ += tmp_correct_
            n_samples_ += len(labels)

            loss.backward()
            if j % batch_accumulation == 0:
                if self._data_set_name == 'vggface2':
                    print('Train [{}] - [{}]/[{}] --- loss: {:.3f} --- acc: {:.2f}% --- curr_index: {} --- lowering_resolution_prob: {}'
                          .format(epoch+1, batch_idx + 1, loader_len, l_ / (batch_idx + 1), (correct_ / n_samples_) * 100, curr_index[0], lowering_resolution_prob[0]))
                else:
                    print('Train [{}] - [{}]/[{}] --- loss: {:.3f} --- acc: {:.2f}%'
                          .format(epoch + 1, batch_idx + 1, loader_len, l_ / (batch_idx + 1), (correct_ / n_samples_) * 100))
                j = 1
                nb_backward_steps += 1
                self._optimizer.step()
                self._optimizer.zero_grad()
                save_stats_on_csv(file_name=self._train_stats_csv,
                                  epoch=epoch,
                                  loss=l_ / (batch_idx + 1),
                                  prec_1=correct_ / float(n_samples_),
                                  lr=get_current_learning_rate(self._optimizer))
            else:
                j += 1
        self._model_hook.remove() if self._model_hook is not None else None

    def _valid_test(self, epoch, test_model=False):

        csv_files = [self._valid_stats_csv, self._valid_stats_orig_csv]

        self._model_manager.set_eval_mode()
        desc = [
            'Model validation on down sampled images',
            'Model validation on original images',
            'Model test'
        ] if self._data_set_name == 'vggface2' else [
                                                 'Model Validation',
                                                 'Model Test'
                                                ]
        with torch.no_grad():

            if test_model:
                loaders_indices = [2]
            else:
                if self._data_set_name == 'vggface2':
                    loaders_indices = [2, 1]
                else:
                    loaders_indices = [1]

            for loader_idx in loaders_indices:
                loader_len, dataset_len = self._data_manager.init_progress_bar_by_index(idx=loader_idx,
                                                                                        pb_description=desc[loader_idx-1],
                                                                                        pb_leave=False)
                local_data_loader = self._data_manager.get_loader_by_index()

                loss_lr = 0.0
                correct_lr = 0.0
                n_samples = 0

                for items in local_data_loader:

                    loss, logits, labels, tmp_correct_, _, _ = self._eval_batch(loader_idx, items)

                    loss_lr += loss.item()
                    correct_lr += tmp_correct_
                    n_samples += len(labels)

                    self._data_manager.update_progress_bar()

                loss_ = float(loss_lr) / loader_len
                acc_ = (float(correct_lr) / n_samples) * 100

                if not test_model:
                    save_stats_on_csv(file_name=csv_files[loader_idx-1], epoch=epoch, loss=loss_, prec_1=acc_,
                                      lr=get_current_learning_rate(self._optimizer))

                self._data_manager.close_progress_bar_by_index()

                if test_model:
                    print('\tTest loss: {:.3f} --- Test acc: {:.2f}%'.format(loss_, acc_))
                else:
                    if self._data_set_name == 'vggface2':
                        if loader_idx == 1:
                            print('\tValid loss down sampled imgs: {:.3f} --- Valid acc down sampled imgs: {:.2f}%'.format(loss_, acc_))
                        else:
                            print('\tValid loss orig imgs:         {:.3f} --- Valid acc orig imgs:         {:.2f}%'.format(loss_, acc_))
                    else:
                        print('\tValid loss: {:.3f} --- Valid acc: {:.2f}%'.format(loss_, acc_))

        return loss_, acc_

    def _extract_features(self):
        self._model_manager.set_eval_mode()
        with torch.no_grad():
            loaders_num, loaders_names = self._data_manager.get_loaders_info()
            print('Total loaders: {} -- Loaders names: {}'.format(loaders_num, loaders_names))
            for loader_idx in range(loaders_num):
                desc = 'Extracting features from "{}" dataset'.format(loaders_names[loader_idx])
                loader_len, dataset_len = self._data_manager.init_progress_bar_by_index(idx=loader_idx,
                                                                                        pb_description=desc,
                                                                                        pb_leave=True)
                local_data_loader = self._data_manager.get_loader_by_index()

                output_f_path = os.path.join(self._outputs_folders, self._time)
                if not os.path.exists(output_f_path):
                    os.makedirs(output_f_path)
                output_f_path = os.path.join(output_f_path, loaders_names[loader_idx])
                if not os.path.exists(output_f_path):
                    os.makedirs(output_f_path)

                label_file = open(os.path.join(output_f_path, 'labels.txt'), 'w')
                print(' ')
                print('Features will be saved here: {}'.format(output_f_path))
                print(' ')
                row_count = 0
                chunk_rows = dataset_len/10

                with h5py.File(os.path.join(output_f_path, 'features.hdf5'), 'a') as hf:
                    dset = hf.create_dataset('data', (dataset_len, 2048), dtype=self._features_type, chunks=(chunk_rows, 2048))

                    for items in local_data_loader:

                        items[0] = items[0].to(device=self._data_manager.get_device(), non_blocking=True)

                        descriptors, logits = self._model_manager.model_forward(items[0])
                        descriptors = descriptors.detach().cpu().numpy()

                        dset[row_count:row_count+descriptors.shape[0]] = descriptors[:]

                        row_count += descriptors.shape[0]

                        [label_file.write(item+'\n') for item in str(items[1])]

                        self._data_manager.update_progress_bar()

                    self._data_manager.close_progress_bar_by_index()

                label_file.close()

    def extract_features_scface(self):
        base = '/mnt/datone/datasets/scface/SCface_database_azh8rd5stx/SCface_database'
        dirs = ['mugshot_frontal_original_all_cropped_by_me', 'surveillance_cameras_all_cropped']
        for dir_ in dirs:
            output_main_folder = os.path.join(base, dir_+'_features')
            if not os.path.exists(output_main_folder):
                os.makedirs(output_main_folder)
            path = os.path.join(base, dir_)
            for img in tqdm(os.listdir(path), total=len(os.listdir(path)), desc="On directory: {}".format(dir_)):
                print("creating file", os.path.join(output_main_folder, img.split('.')[0]+'.hdf5'))
                img_path = os.path.join(path, img)
                print("image", img_path)
                with h5py.File(os.path.join(output_main_folder, img.split('.')[0]+'.hdf5'), 'w') as hf:
                    dset = hf.create_dataset('data', (1, 2048), dtype=self._features_type)
                    img = get_img_tensor(img_path)[np.newaxis, :]
                    descriptors, logits = self._model_manager.model_forward(img)
                    descriptors = descriptors.detach().cpu().numpy()
                    dset[0] = descriptors
    
    
