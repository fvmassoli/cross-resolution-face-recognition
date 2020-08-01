import numpy as np
from tqdm import tqdm

import torch.nn.functional as F

from utils import *


class RunManager(object):
    def __init__(self, student, teacher, optimizer, scheduler, data_manager, dataset, lowering_resolution_prob,
                device, curriculum, epochs, batch_accumulation, lambda_, train_steps, run_mode, 
                super_resolved_images, out_dir, logging):
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
        self._batch_accumulation = batch_accumulation
        self._lambda = lambda_
        self._train_steps = train_steps
        self._run_mode = run_mode
        self._super_resolved_images = super_resolved_images
        self._out_dir = out_dir
        self._logging = logging
        self._features_type = np.float32

    def _eval_batch(self, loader_idx, data):
        curr_index = -1
        lowering_resolution_prob = -1
        items = [item for item in data]

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
        student_features, student_logits = self._student(batch)

        y_hat = student_logits.argmax(dim=1)
        correct_lr = (y_hat == labels).sum().item()
        
        loss = F.cross_entropy(student_logits, labels) + self._lambda*F.mse_loss(student_features, teacher_features)

        return loss, student_logits, labels, correct_lr, curr_index, lowering_resolution_prob

    def _train(self, epoch):
        self._student.train()
        self._optimizer.zero_grad()

        loader_len, dataset_len = self._data_manager.init_progress_bar_by_index(
                                                                            idx=0,
                                                                            pb_description='Training',
                                                                            pb_leave=False
                                                                        )
        local_data_loader = self._data_manager.get_loader_by_index()

        j = 1
        loss_ = 0
        best_acc = 0
        correct_ = 0
        n_samples_ = 0
        nb_backward_steps = 0
        
        self._logging.info("#"*30)
        self._logging.info(f'Training at epoch: {epoch}')
        self._logging.info("#"*30)

        for batch_idx, data in enumerate(local_data_loader, 1):

            if nb_backward_steps == self._train_steps:
                nb_backward_steps = 0
                self._student.eval()
                v_l_, tmp_best_acc = self._val(epoch)
                self._student.train()
                self._scheduler.step(v_l_, epoch+1)
                ## Save best model
                if tmp_best_acc > best_acc:
                    best_acc = tmp_best_acc
                    save_model_checkpoint(
                                    best_acc, 
                                    batch_idx, 
                                    epoch, 
                                    self._student.get_model_state_dict(), 
                                    self._out_dir, 
                                    self._logging
                                )
                
            loss, logits, labels, tmp_correct_, curr_index, lowering_resolution_prob = self._eval_batch(loader_idx=0, data=data)

            loss_ += loss.item()
            correct_ += tmp_correct_
            n_samples_ += len(labels)

            loss.backward()
            if j % self._batch_accumulation == 0:
                self._logging.info(
                            f'Train [{epoch}] - [{batch_idx}]/[{loader_len}] --- '
                            f'loss: {loss_/batch_idx:.3f} --- acc: {(correct_/n_samples_)*100:.2f}% --- '
                            f'curr_index: {curr_index[0]} --- lowering_resolution_prob: {lowering_resolution_prob[0]}'
                        )
                j = 1
                nb_backward_steps += 1
                self._optimizer.step()
                self._optimizer.zero_grad()
            else:
                j += 1
        
    def _val(self, epoch, test_model=False):

        csv_files = [self._valid_stats_csv, self._valid_stats_orig_csv]
        desc = [
            'Model validation on down sampled images',
            'Model validation on original images',
            'Model test'
        ] if self._dataset_path == 'vggface2' else [
                                                 'Model Validation',
                                                 'Model Test'
                                                ]
        with torch.no_grad():

            if test_model:
                loaders_indices = [2]
            else:
                if self._dataset_path == 'vggface2':
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
                    if self._dataset_path == 'vggface2':
                        if loader_idx == 1:
                            print('\tValid loss down sampled imgs: {:.3f} --- Valid acc down sampled imgs: {:.2f}%'.format(loss_, acc_))
                        else:
                            print('\tValid loss orig imgs:         {:.3f} --- Valid acc orig imgs:         {:.2f}%'.format(loss_, acc_))
                    else:
                        print('\tValid loss: {:.3f} --- Valid acc: {:.2f}%'.format(loss_, acc_))

        return loss_, acc_

    def run(self):
        assert self._dataset == 'vggface2' or self._dataset == 'vggface2-500', f"Training not implemented for dataset: {self._dataset}"
        self._val(0)
        [self._train(epoch) for epoch in range(1, self._epochs+1)]
        if self._dataset == 'vggface2-500':
            self._val(0, test_model=True)
        
    def extract_features(self):
        self._student.eval()
        with torch.no_grad():
            loaders_num, loaders_names = self._data_manager.get_loaders_info()
            self._logging.info(f'Total loaders: {loaders_num} -- Loaders names: {loaders_names}')
            for loader_idx in range(loaders_num):
                desc = 'Extracting features from "{}" dataset'.format(loaders_names[loader_idx])
                loader_len, dataset_len = self._data_manager.init_progress_bar_by_index(idx=loader_idx,
                                                                                        pb_description=desc,
                                                                                        pb_leave=True)
                local_data_loader = self._data_manager.get_loader_by_index()
                
                output_f_path = os.path.join(self._out_dir, 'features')
                if not os.path.exists(output_f_path):
                    os.makedirs(output_f_path)

                label_file = open(os.path.join(output_f_path, 'labels.txt'), 'w')
                self._logging.info(f"Features will be saved here: {output_f_path}")
                row_count = 0
                chunk_rows = dataset_len/10

                with h5py.File(os.path.join(output_f_path, 'features.hdf5'), 'a') as hf:
                    dset = hf.create_dataset('data', (dataset_len, 2048), dtype=self._features_type, chunks=(chunk_rows, 2048))

                    for items in local_data_loader:

                        items[0] = items[0].to(device=self._data_manager.get_device(), non_blocking=True)

                        descriptors, _ = self._student(items[0])
                        descriptors = descriptors.detach().cpu().numpy()

                        dset[row_count:row_count+descriptors.shape[0]] = descriptors[:]

                        row_count += descriptors.shape[0]

                        [label_file.write(item+'\n') for item in str(items[1])]

                        self._data_manager.update_progress_bar()

                    self._data_manager.close_progress_bar_by_index()

                label_file.close()



    # def extract_features_scface(self):
    #     base = '/mnt/datone/datasets/scface/SCface_database_azh8rd5stx/SCface_database'
    #     dirs = ['mugshot_frontal_original_all_cropped_by_me', 'surveillance_cameras_all_cropped']
    #     for dir_ in dirs:
    #         output_main_folder = os.path.join(base, dir_+'_features')
    #         if not os.path.exists(output_main_folder):
    #             os.makedirs(output_main_folder)
    #         path = os.path.join(base, dir_)
    #         for img in tqdm(os.listdir(path), total=len(os.listdir(path)), desc="On directory: {}".format(dir_)):
    #             print("creating file", os.path.join(output_main_folder, img.split('.')[0]+'.hdf5'))
    #             img_path = os.path.join(path, img)
    #             print("image", img_path)
    #             with h5py.File(os.path.join(output_main_folder, img.split('.')[0]+'.hdf5'), 'w') as hf:
    #                 dset = hf.create_dataset('data', (1, 2048), dtype=self._features_type)
    #                 img = get_img_tensor(img_path)[np.newaxis, :]
    #                 descriptors, logits = self._model_manager.model_forward(img)
    #                 descriptors = descriptors.detach().cpu().numpy()
    #                 dset[0] = descriptors
    
    
