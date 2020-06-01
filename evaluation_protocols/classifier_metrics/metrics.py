from __future__ import division

import os
import numpy as np
from tqdm import tqdm

from sklearn import metrics
from scipy.interpolate import interp1d


def eval_roc(scores_dict, resolutions, model=None, template_matching=None):
    """

    Evaluates the ROC for each model at each resolution

    :param scores_dict: dict of scores, k: resolution, v: scores
    :param resolutions: list of resolution values
    :param template_matching: list of templates to match and the corresponding labels
    :param model: model name (used only for the print statement)

    :return: numpy array of (fpr, tpr, thr)
    """

    return np.asarray([metrics.roc_curve(template_matching, scores_dict[res], pos_label=1) for res in resolutions])


def get_cross_resolution_scores(models_base_path, model, data_set_name, date, template_matching, far=1.e-3):
    """

    Evaluates TAR @ FAR

    :param models_base_path: base path to models
    :param model: model name
    :param data_set_name: data set nema
    :param date: date of the data generation
    :param template_matching: list of template matching and relative labels
    :param far: far value at which evaluate the tar

    :return: TAR @ FAR
    """

    print('Loading scores...')
    path = os.path.join(models_base_path, model, data_set_name, date, 'scores/scores_cross_correlation')
    cross_correlation_scores = np.load(os.path.join(path, 'similarity_scores.npy'))
    print('Scores loaded!!!')

    results = []
    for i in tqdm(range(len(cross_correlation_scores)), total=len(cross_correlation_scores), desc='Looping!!!!'):

        tmp = []
        for j in range(len(cross_correlation_scores[i])):

            (fpr, tpr, thr) = metrics.roc_curve(template_matching, cross_correlation_scores[i][j][:], pos_label=1)
            r_f = interp1d(fpr, tpr)
            tmp.append(round(r_f(far) * 100., 2))

        results.append(np.asarray(tmp))

    return np.asarray(results)


def eval_cmc(probe_features_norm, probe_subj_ids, gallery_features_norm, gallery_subj_ids, mAP=False):
    """

    Eval the Cumulative Match Characteristics for the 1:N Face Identification protocol (close set scenario)

    :param probe_features_norm: normalized probe features
    :param probe_subj_ids: subject ids of probe features
    :param gallery_features_norm: normalized gallery features
    :param gallery_subj_ids: subject ids of gallery features
    :param mAP: boolean. If True skip CMC final evaluation

    :return: points: ranks at which the CMC has been evaluated, retrieval_rates: CMC for each rank value
    """

    if len(gallery_subj_ids.shape) == 1:
        gallery_subj_ids = gallery_subj_ids[:, np.newaxis]
    if len(probe_subj_ids.shape) == 1:
        probe_subj_ids = probe_subj_ids[:, np.newaxis]

    print('\t\tEvaluating distance matrix...')

    distance_matrix = np.dot(probe_features_norm, gallery_features_norm.T)
    ranking = distance_matrix.argsort(axis=1)[:, ::-1]
    ranked_scores = distance_matrix[np.arange(probe_features_norm.shape[0])[:, np.newaxis], ranking]
    print('\t\tDistance matrix evaluated!!!')

    gallery_ids_expanded = np.tile(gallery_subj_ids, probe_features_norm.shape[0]).T
    gallery_ids_ranked = gallery_ids_expanded[np.arange(probe_features_norm.shape[0])[:, np.newaxis], ranking]

    ranked_gt = (gallery_ids_ranked == probe_subj_ids).astype(np.int8)

    nb_points = 50
    points = np.arange(1, nb_points+1)
    retrieval_rates = np.empty(shape=(nb_points, 1))

    if not mAP:
        for k in points:
            retrieval_rates_ = ranked_gt[:, :k].sum(axis=1)
            retrieval_rates_[retrieval_rates_ > 1] = 1
            retrieval_rates[k - 1] = np.average(retrieval_rates_)

    return points, retrieval_rates, ranked_scores, ranked_gt


def eval_fpir(unmated_query_indeces, distance_matrix):
    """

    Eval FPIR: False Positive Identification Rate.
               How many unmated probe has a score higher than a specific threshold

    :param unmated_query_indeces: list of indeces (indices of the distance_matrix's rows) corresponding to unmated probes subject ids
    :param distance_matrix: matrix of distances among features

    :return: thresholds, FPIR
    """

    highest_scores = []

    for i, n in enumerate(unmated_query_indeces):
        # n is the index of the query that has not a mate into the gallery
        # get the distances of a single query from all templates
        score = distance_matrix[n]

        # sort in descending order
        score = np.sort(score)[::-1]

        # only consider the highest score (the most similar retrieved face)
        highest_scores.append(score[0])

    # the searching step
    highest_scores = np.asarray(highest_scores)
    min_ = np.min(highest_scores)
    max_ = np.max(highest_scores)
    step = (max_ - min_) / 1000.
    thresholds = []
    FPIRs = []

    for thr in np.logspace(-4, 0, 4000): #np.arange(min_, max_, step):
        # for each value of the thresholds counts how many queries returned the most similar
        # face above the threshold. In this case we expect the curve to go down as fast as possible
        # since all the queries do not belong to any of the templates
        current_fpir = np.sum((highest_scores > thr).astype(np.uint8)) / highest_scores.size

        # the thresholds will be used when calculate the corresponding FNIR
        thresholds.append(thr)
        FPIRs.append(current_fpir)

    return thresholds, np.asarray(FPIRs)


def eval_fnir(mated_indeces, distance_matrix, thresholds, probe_subject_id, gallery_subject_id, r=20):
    """

    Eval FNIR: False Negative Identification Rate.
               How many mated matches, within a certain rank, are below a certain threshold.

    :param mated_indeces: indeces in the distance matrix of the query that have mated into the gallery. Indeces of the
                          query subject id that have a mate into the gallery
    :param distance_matrix:
    :param thresholds:
    :param probe_subject_id:
    :param gallery_subject_id:
    :param r: highest rank among which look for results

    :return:
    """

    gt_scores = []
    for p, mi in enumerate(mated_indeces):

        # subject id of the query that has a mate into the gallery
        mated_id = probe_subject_id[mi]

        # lista di indici
        good_index = np.where(gallery_subject_id == mated_id)[0]

        score = distance_matrix[mi]

        rank = np.argsort(score)[::-1]

        flag = 0

        for i in range(r):
            find_ = np.where(good_index == rank[i])[0]
            if len(find_) != 0:
                gt_scores.append(score[rank[find_[0]]])
                flag = 1
                break

        if flag == 0:
            gt_scores.append(0.0)  # matching fails

    gt_scores_ar = np.asarray(gt_scores)
    FNIRs = []

    for thr in thresholds:
        curr_fnir = np.sum((gt_scores_ar < thr).astype(np.uint8)) / gt_scores_ar.size
        FNIRs.append(curr_fnir)

    return thresholds, np.asarray(FNIRs)
