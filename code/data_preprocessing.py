# Loading data;
## get_features
## load_data
# Centering data;
# Finding + completing missing data.
# stacking data

from numpy.random import default_rng
import scipy.io as sio
import numpy as np

import linear_subspaces


def load_data(file): return sio.loadmat(file)


def get_features(data, feature, from_ix, to_ix):
    # visual properties are encoded here, being the features a mapping of the images (array of 600k RGB) to a 512 vector 
    # D' has dimensions of 600k x 10482, being necessarry more memory for processing this data and represent. 
    # D has dimensions of 512 x 10482, encoding the visual properties 

    # this mapping cleans the output, in the sense that things that change a bit have teh same output, 
    # se não fosse assim o rank seria necessariamente maior 

    features = data[feature]
    shape = features.shape
    print("Data has got {1} {0}-sized objects.".format(shape[0], shape[1]))  # change to log
    return features, features[:, from_ix:to_ix + 1]


def RANSAC(features, rank = 10, k=1000, t=0.7, ord=2):
    #k – Maximum number of iterations allowed in the algorithm.
    #t – Threshold value to determine data points that are fit well by model
    
    rng = default_rng()
    
    #best_error = np.inf 

    num_inliers = 0
    
    for _ in range(k):
        
        ids = rng.permutation(features.shape[1])
    
        maybe_inliers = ids[: rank] # randomly selected frames from data

        # model parameters fitted to maybe_inliers
        basis,sigma ,v_T = linear_subspaces.factorisation(features[:, maybe_inliers], rank)
        
        projector, null_projector = linear_subspaces.get_basis_projectors(basis)
        
        proj = linear_subspaces.project(projector, features)
        
        orth_proj = linear_subspaces.project(null_projector, features)
        
        norm_proj = np.linalg.norm(proj, ord = ord, axis=0)/np.linalg.norm(features, ord = ord, axis=0)
        
        norm_orth_proj = np.linalg.norm(orth_proj, ord = ord, axis=0)/np.linalg.norm(features, ord = ord, axis=0)
             
        also_inliers = np.argwhere(norm_proj > t)
        outliers = np.argwhere(norm_orth_proj > t)
        #np.delete(np.arange(0, features.shape[1]), also_inliers)

        if also_inliers.size > num_inliers:
            num_inliers = also_inliers.size
            best_fit = (basis, sigma, v_T)

    return best_fit, outliers



def filter_dataset(dataset, prob_tolerance, num_joint_tolerance):
    """ Joints with less than p = tolerance probability of detection are 
    considered badly read - as bad as having missing data."""

    # dataset = np.array([[1,2,0], [1,0,0], [0,1,1], [3,5,1], [3,5,0.1], [1,0.1,0.1]])
    # assert dataset.shape[0] == 3 * setup.NUM_JOINTS

    def _mask_bad_joints(dataset, tolerance):
        mask = np.zeros_like(dataset)
        mask[2::3] = 1

        probs = mask * dataset
        mask = np.where(probs < tolerance, 0, mask)

        bad_entries = dataset[2::3].size - np.count_nonzero(mask)
        n_total_joints = dataset.shape[1] * dataset.shape[0] / 3
        percentage = bad_entries / n_total_joints
        print('There are {0} entries - {1:.4g}% - with a probability of detection lower than {2}%.'.format(bad_entries, 100 * percentage, 100 * tolerance))

        mask[::3] = mask[2::3]
        mask[1::3] = mask[2::3]

        return mask * dataset, percentage

    def _remove_bad_skeletons(dataset, tolerance):
        probs = dataset[2::3]
        mask = np.where(probs == 0, 0, 1)

        n_joints_per_skel = np.count_nonzero(mask, axis=0)
        print('Number of good joints per skeleton: ', n_joints_per_skel)

        bad_skels = np.argwhere(n_joints_per_skel <= tolerance)
        percentage = bad_skels.size / dataset.shape[1]

        print('There are {0} skeletons - {1:.4g}% - with not enough joints, whose indexes are: '.format(bad_skels.size, 100 * percentage), bad_skels)
        dataset = np.delete(dataset, bad_skels, axis=1)
        return dataset, percentage

    prev_num_skeletons = dataset.shape[1]
    print('The dataset has got {0} skeletons.'.format(prev_num_skeletons))

    dataset, bad_joint_percentage = _mask_bad_joints(dataset, prob_tolerance)
    dataset, bad_skels_percentage = _remove_bad_skeletons(dataset, num_joint_tolerance)

    new_num_skeletons = dataset.shape[1]
    print('The new dataset has now got {0} skeletons - {1:.4g}% of the original.'.format(new_num_skeletons, 100 * new_num_skeletons / prev_num_skeletons))

    return dataset, bad_joint_percentage, bad_skels_percentage


def find_missing_data(incomplete_data):
    differences = incomplete_data.astype(bool)
    num_entries = differences.size
    missing_entries = differences.size - np.count_nonzero(differences)
    percentage = missing_entries / num_entries if num_entries != 0 else 0
    print('There are {0} - {1:.5g}% - missing entries in the incomplete dataset.'.format(missing_entries, 100 * percentage))

    differences_column = np.all(differences, axis=0)
    different_columns = np.argwhere(differences_column == False).reshape(-1)

    # There are 17591 incomplete columns on the whole dataset
    print('There are {0} incomplete columns.'.format(different_columns.size))

    equal_columns = np.delete(np.arange(incomplete_data.shape[1]), different_columns)

    # There are 3202 complete columns on the whole dataset
    print('There are {0} complete columns.'.format(equal_columns.size))

    differences_row = np.all(differences, axis=1)
    different_rows = np.argwhere(differences_row == False).reshape(-1)
    print('There are {0} incomplete rows.'.format(different_rows.size))

    equal_rows = np.delete(np.arange(incomplete_data.shape[0]), different_rows)
    print('There are {0} complete rows.'.format(equal_rows.size))

    return differences, different_columns, equal_columns, different_rows, equal_rows, percentage


def center_data(data, mean):
    return data - mean


def dataset_stacking(skeletons, features):
    frames_skeletons = skeletons[0, :]

    frames_no_skeletons = np.setdiff1d(np.arange(0, features.shape[1]), frames_skeletons)

    frames_stacked = sorted(np.append(frames_skeletons, frames_no_skeletons))

    stacked_matrix = np.zeros((skeletons.shape[0] + features.shape[0] + 1, len(frames_stacked)))
    # stacked_matrix = np.empty((skeletons.shape[0] + features.shape[0], len(frames_stacked)))
    stacked_matrix[0, :] = frames_stacked

    for i in range(len(frames_stacked)):

        if frames_stacked[i] in frames_skeletons:
            stacked_matrix[2:skeletons.shape[0] + 1, i] = skeletons[1:, int(frames_stacked[i])]

            # stacked_matrix[1, i] = frames_stacked.count(frames_stacked[i])

            stacked_matrix[1, i] = 1
            # variable that tells if the openpose algorithm detected any skeleton

        else:
            stacked_matrix[1, i] = 0

        stacked_matrix[skeletons.shape[0] + 1:, i] = features[:, int(frames_stacked[i])]

    return stacked_matrix


def complete_missing_data(incomplete_dataset, rank, tolerance, max_iterations):
    def _init(incomplete_dataset):
        """ Substitutes missing values."""
        # may pick other strategies: random, average of each line, 0, ...
        mask, incomplete_col_idxs, _, incomplete_row_idxs, _, _ = find_missing_data(incomplete_dataset)
        replacement = (np.sum(incomplete_dataset, axis=1) / np.count_nonzero(incomplete_dataset, axis=1)).reshape(-1, 1)
        return mask.astype(int), np.where(incomplete_dataset > 0, incomplete_dataset, replacement)

    def _input_observations(observations, opposite_mask, approximation):
        return observations + opposite_mask * approximation

    def _error(original, approximation, mask, original_norm):
        """ Computes the error strategy."""
        error = np.linalg.norm(mask * (original - approximation), ord='fro')
        return 1 - error / original_norm

    mask, data_fill = _init(incomplete_dataset)

    opposite_mask = (np.ones_like(mask) - mask)
    iteration = data_fill
    approximation = data_fill
    num_iteration = 0

    gap = 0
    init_norm = np.linalg.norm(incomplete_dataset)

    while gap < tolerance and num_iteration < max_iterations:
        num_iteration += 1
        # iteration = incomplete_dataset + opposite_mask * approximation
        iteration = _input_observations(incomplete_dataset, opposite_mask, approximation)

        U_k, sigma_k, V_t_k = linear_subspaces.factorisation(iteration, rank)
        sigma_k = np.diag(sigma_k[:rank])
        approximation = U_k @ sigma_k @ V_t_k

        gap = _error(incomplete_dataset, approximation, mask, init_norm)

    print('Completed after {0} iterations, being {1:.7g}% close to the original matrix.'.format(num_iteration, 100 * gap))
    return _input_observations(incomplete_dataset, opposite_mask, approximation), gap


if __name__ == '__main__':
    import plotting, setup

    filtered_dataset, bad_joint_percentage, bad_skel_percentage = filter_dataset(setup.INCOMPLETE_SKELS_DEV, setup.JOINT_PERCENT_TOLERANCE, setup.NUM_JOINTS_CUTOFF)
    plotting.pie_plot((('Below threashold', bad_joint_percentage), ('Above threashold', 1 - bad_joint_percentage)),
                      'Number of joints above {0}% threshold'.format(100 * setup.JOINT_PERCENT_TOLERANCE))

    _, _, _, _, _, missing_data_percentage = find_missing_data(filtered_dataset)
    plotting.pie_plot((('Missing data', missing_data_percentage), ('Complete data', 1 - missing_data_percentage)), 'Missing data ratio')
