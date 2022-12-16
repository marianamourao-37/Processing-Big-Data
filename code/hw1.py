import matplotlib.pyplot as plt
import matplotlib.cm as pltc
import scipy.io as sio
import numpy as np
import skimage
import cv2

## Support Functions

# Printed information

def feat_info(data):
    print(f'Data has {data.shape[1]} frames with {data.shape[0]} features each.')

def skel_info(data):
    last = int(data[0][-1])
    frames = [i for i in range(last + 1)]
    hist = [0] * (last + 1)
    for i in data[0]:
        hist[int(i)] += 1
    # bar_plot(frames, hist, 'Skels per frame', 'Frame', '# of Skels')
    plt.plot(hist)
    plt.show()

def video_info(video):
    print(f'Size: {int(video.get(3))} * {int(video.get(4))}')
    print(f'FPS: {video.get(5)}')
    print(f'#Frames: {int(video.get(7))}')

def play(video):
    video_info(video)
    while (video.isOpened()):
        ret, frame = video.read()
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
    video.release()
    cv2.destroyAllWindows()

# Frame manipulation

def select_frames(video, frames):
    selected = []
    for frame_number in frames:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        selected += [cv2.cvtColor(video.read()[1], cv2.COLOR_BGR2RGB)]
    return selected

def montage(frames):  # This bitch stopped working wtf
    skimage.util.montage(frames)

def extract_frames(video, frames, nrows_plot, ncols_plot, title_plot, save=False, to_path='./'):
    fig, axs = plt.subplots(nrows_plot, ncols_plot, figsize=(15, 12))
    fig.suptitle(title_plot + '\n', fontsize=24)
    plt.tight_layout()
    for num_frame, ax in zip(range(len(frames)), axs.ravel()):
        video.set(cv2.CAP_PROP_POS_FRAMES, frames[num_frame]);
        ret, frame = video.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if ret:
            ax.set_title('frame ' + str(frames[num_frame] + 1))
            ax.imshow(frame)
            ax.axis('off')
            if save:
                cv2.imwrite(to_path + 'frame_' + str(frames[num_frame]) + '.png', frame)
    plt.show()
    video.release()
    cv2.destroyAllWindows()

# Plotting


def bar_plot(x, y, title, xlabel, ylabel):
    plt.figure(figsize=(15, 6), tight_layout=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    margin = 0.05 * (max(y) - min(y))
    plt.gca().set_ylim([min(y) - margin, max(y) + 2 * margin])
    colours = pltc.turbo(plt.Normalize(min(y), max(y))(y))
    plt.bar(x, y, width=1, align='edge', color=colours)
    plt.show()

def scatter_plot(coefficients, title, xlabel, ylabel, labels):
    x = coefficients[0]
    y = coefficients[1]

    plt.title(title)
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    for xi, yi, label in zip(x, y, labels):
        plt.annotate(label,  # this is the text
                     (xi, yi),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(5, 10),  # distance from text to points (x,y)
                     ha='center')

    plt.show()

def connected_scatter_plot(coefficients, title, xlabel, ylabel):
    x = coefficients[0]
    y = coefficients[1]

    pairs_x_consecutive = [[x[i], x[i + 1]] for i in range(len(x) - 1)]
    pairs_y_consecutive = [[y[i], y[i + 1]] for i in range(len(y) - 1)]

    plt.title(title)

    plt.plot(pairs_x_consecutive, pairs_y_consecutive, 'b-', label="consecutive images linkage")
    plt.plot(x, y, 'r*')
    plt.legend()

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.show()

def plot_sigularvalues(fact, logplot=False):
    sigma = fact[1]
    R = sigma.shape[0]

    plt.figure()
    plt.title('Singular values')
    plt.plot(sigma)
    plt.grid()
    plt.show()

    if logplot:
        plt.figure()
        plt.title('log singular values')
        plt.plot(np.log(sigma))
        plt.grid()
        plt.show()

    cumsum_sigma1 = [np.sum(sigma[k:R]) for k in range(R)]

    cumsum_sigma2 = [np.sum(sigma[:k]) for k in range(R)]

    plt.figure()
    plt.plot(cumsum_sigma1 / np.sum(sigma), 'b')
    plt.plot(cumsum_sigma2 / np.sum(sigma), 'r')
    plt.grid()
    plt.show()

## Main Functions

def load_data(dataset):
    if dataset == 'rai':
        d = sio.loadmat(f'../data/{dataset}/skel_all.mat')
        t, poses, poses_complete, p = d['timestamp'], d['poses'], d['poses_complete'], d['prob']

        poses = np.concatenate([np.concatenate((poses[2 * i:2 * i + 2], p[i:i + 1])) for i in range(p.shape[0])])

        poses_complete = np.concatenate([np.concatenate((poses_complete[2 * i:2 * i + 2], p[i:i + 1])) for i in range(p.shape[0])])

        return np.concatenate((t, poses)), \
               np.concatenate((t, poses_complete)), \
               None, \
               cv2.VideoCapture(f'../data/{dataset}/video.avi')

    return sio.loadmat(f'../data/{dataset}/skel.mat')['skeldata'], \
           sio.loadmat(f'../data/{dataset}/skel_complete.mat')['skeldata'], \
           sio.loadmat(f'../data/{dataset}/feat.mat')['features'], \
           cv2.VideoCapture(f'../data/{dataset}/video.mp4')

def sub_data(data, first_frame, last_frame):
    return data[:, first_frame - 1:last_frame]

def SVD(data, rank=0):
    D = np.linalg.svd(data)
    if rank != 0:
        u = D[0][:, :rank]
        s = D[1][:rank]
        v = D[2][:rank, :]
        return u, s, v
    return D

def singular_values_analysis(fact, threshold):
    def representativity_values(sigma, index=-1):
        representativity = np.cumsum(sigma) / np.sum(sigma) * 100
        if index != -1: return representativity[index]
        return representativity

    sigma = fact[1]
    representativity = representativity_values(sigma)
    plt.title(f'Singular values - weights of the U vectors basis size {sigma.size}')
    plt.plot(representativity, sigma, 'b.-')
    plt.xlabel('% of Representativity')
    plt.ylabel('Singular values')
    plt.axvline(x=threshold, color='red', linestyle='dotted', linewidth=2)

    min_repr = np.argmax(representativity > threshold)
    sigma_min_repr = fact[1][min_repr]

    cumulative_repr = representativity_values(sigma, min_repr)

    plt.arrow(threshold + 5.5, sigma_min_repr + 5.5, -2, -2, width=0.5, color='black')
    plt.annotate('{0:.3g}%'.format(cumulative_repr),
                 (threshold + 7, sigma_min_repr + 5.5),
                 textcoords="offset points",
                 xytext=(5, 5),
                 ha='center')
    plt.show()
    print('We can get {0:.5g}% representativity with {1} singular values.'.format(cumulative_repr, min_repr + 1))

# dataset = 'all'
# # dataset = 'cut'
# # dataset = 'rai'
# #
# skeli, skelc, feat, video = load_data(dataset)
#
# THRESHOLD = 90
#
# # Full features' data
#
# feat_fact = SVD(feat)
# feat_fact_r2 = SVD(feat, 2)
# feat_coef_r2 = feat_fact_r2[0].T @ feat
#
# connected_scatter_plot(feat_coef_r2, f'Distribution of full dataset of dimension {feat.shape[0]} on rank = 2 subspace', 'x1', 'x2')
# plot_sigularvalues(feat_fact, True)
# singular_values_analysis(feat_fact, THRESHOLD)
#
# # Subspace (12)
#
# FRAME_I = 5895
# FRAME_F = 5906
# LABELS = np.arange(FRAME_I, FRAME_F + 1)
#
# feat12 = sub_data(feat, FRAME_I, FRAME_F)
#
# feat12_fact = SVD(feat12)
# feat12_fact_r2 = SVD(feat12, 2)
# feat12_coef_r2 = feat12_fact_r2[0].T @ feat12
#
# scatter_plot(feat12_coef_r2, f'Distribution of images {FRAME_I}-{FRAME_F} on 2 dimensions', 'x1', 'x2', LABELS)
# plot_sigularvalues(feat12_fact, True)
# singular_values_analysis(feat12_fact, THRESHOLD)

###### FUCK IT I STOPPED HERE, WE'LL CONTINUE LATER ######

# def projectors(fact):
#     b = fact[0]
#     b_projector = b @ b.T
#     return b_projector, \
#            (np.eye(b.shape[0]) - b_projector)
#
#
# def project(projector, projected):
#     return projector @ projected
#
#
# def compare_projections(features, fact, keep_first_n):
#     projector, null_projector = projectors(fact)
#
#     proj = projector @ features
#     orth_proj = null_projector @ features
#
#     print("All original points are close to proj + orth_proj: {0}".format(np.isclose(features, proj + orth_proj).all()))
#
#     norm_proj = np.linalg.norm(proj, axis=0) / np.linalg.norm(features, axis=0)
#     norm_orth_proj = np.linalg.norm(orth_proj, axis=0) / np.linalg.norm(features, axis=0)
#
#     idx_larger_projnorm = np.argsort(-norm_proj)[0:keep_first_n]
#     idx_larger_orthprojnorm = np.argsort(-norm_orth_proj)[0:keep_first_n]
#
#     return (norm_proj, norm_orth_proj), (idx_larger_projnorm, idx_larger_orthprojnorm)
#
#
# top_projections = 100
# (norm_proj, norm_orth_proj), (idx_larger_projnorm, idx_larger_orthprojnorm) = compare_projections(feat, feat12_fact, top_projections)
#
# # print('{0} images whose projection on the subspace have got larger norm:\n'.format(top_projections), idx_larger_projnorm)
# # extract_frames(video, idx_larger_projnorm, 10, 10, 'projection on the subspace of rank = 12')
#
#
# # print('{0} images whose projection on its kernel (null) have got larger norm:\n'.format(top_projections), idx_larger_orthprojnorm)
# # extract_frames(video, idx_larger_orthprojnorm, 10, 10, 'projection on the null in relation to the subspace of rank = 12')
#
# extract_frames(video, LABELS, 10, 10, 'projection on the null in relation to the subspace of rank = 12')
