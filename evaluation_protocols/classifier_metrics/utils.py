from __future__ import division

import os
import h5py
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm


def pickle_unpickle_file(model_folder, data_set_name, date, file_name=None, obj=None, save=False, load=False):

    analysis_output_folder = 'analysis_outputs_new_output_features'
    output_path = os.path.join(analysis_output_folder, model_folder, data_set_name, date)

    f_name = os.path.join(output_path, file_name)

    if save:

        assert obj is not None, 'Object to be saved is None!!!'

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        outfile = open(f_name, 'wb')
        pickle.dump(obj, outfile)
        print('Saved file at: {}'.format(f_name))
        outfile.close()

    if load:

        infile = open(f_name, 'rb')
        d = pickle.load(infile)
        infile.close()

        return d


def normalize_features(v):
    return v / np.sqrt(np.sum(v**2, axis=-1, keepdims=True))


def load_templates(meta_folder_path, data_set_name):
    """

    Load template ids for matching and relative labels

    :param meta_folder_path: path to meta folder
    :param data_set_name: data set name

    return: list of template matching and relative labels
    """

    file_name = 'ijbb_template_pair_label.txt' if data_set_name == 'ijbb' else 'ijbc_template_pair_label.txt'
    tmp_ = 'ijbb_evaluation' if data_set_name == 'ijbb' else 'ijbc_evaluation'

    pair_file = open(os.path.join(meta_folder_path, tmp_, 'verification/meta', file_name), 'r')

    template_matching = np.asarray([int(line.split(' ')[-1]) for line in pair_file.readlines()]).astype(np.int8)

    pair_file.close()

    return template_matching


def load_scores(models_base_path, data_set_name, model, date, resolutions):
    """

    Evaluates the scores for each model at each resolution

    :param models_base_path: path to model folder
    :param data_set_name: name of the data set
    :param date: date of the data generation
    :param model: model name
    :param resolutions: list of resolution values

    return: dict of model scores for each resolution; k: resolution, v: scores
    """

    f = os.path.join(models_base_path, model, data_set_name, date, 'scores')

    scores_dict = {res: np.load(os.path.join(f, str(res), 'similarity_scores.npy')) for res in resolutions}

    return scores_dict


def get_data_frames(cross_resolution_scores, resolutions):
    """

    Create data frame for cross resolution face verification

    :param cross_resolution_scores: list of cross resolution scores, one for each resolution
    :param resolutions: list of resolution values

    :return:
    """

    cols = ['res ' + str(res) for res in resolutions]
    index = cols
    rows = []

    for i in range(len(resolutions)):
        rows.append([val for val in cross_resolution_scores[i]])

    return pd.DataFrame(data=rows, columns=cols, index=index)


################################
###### Used for CMC - DET ######
################################
def load_templates_features(features_folder, resolution, data_set_name, date, model, uq_templates_id):
    """

    Load template features following the order from uq_templates_id

    :param features_folder: path to template features
    :param resolution: resolution of the templates
    :param data_set_name: data set name
    :param date: date of the template extraction
    :param model: model name
    :param uq_templates_id: list of unique templates id

    :return: numpy array containing normalized features templates
    """

    template_path = os.path.join(features_folder, model, data_set_name, date, 'template', str(resolution))

    templates = np.empty(shape=(len(uq_templates_id), 2048))

    pb = tqdm(enumerate(uq_templates_id), total=len(uq_templates_id), desc='Loading features', leave=False)
    for c, id_ in pb:
        pb.update(1)
        if not os.path.exists(os.path.join(template_path, 'template_' + str(id_) + '.npy')):
            print('Path not found at: {}'.format(os.path.join(template_path, 'template_' + str(id_) + '.npy')))
        else:
            templates[c] = normalize_features(np.load(os.path.join(template_path, 'template_' + str(id_) + '.npy')))
    pb.close()

    return templates


def load_features_from_h5py(features_base_path, folder_name, data_set_name='data', file_name='features.hdf5'):
    """

    Load features from h5py file

    :param features_base_path: base path to features folder
    :param folder_name: features folder
    :param data_set_name: name of the h5py data set
    :param file_name: name of the features file

    :return: numpy array of normalized features
    """

    return normalize_features(h5py.File(os.path.join(features_base_path, folder_name, file_name), 'r')[data_set_name][()])
