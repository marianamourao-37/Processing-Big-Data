# bar plots
# histograms
# that one graph with all the edges connected
# those 3 graphs belonging to hw1 + singular values graph (elbow)
# heatmaps
# scatters

import matplotlib.pyplot as plt
import matplotlib.cm as pltc
import seaborn as sns
import numpy as np
from IPython.display import clear_output
import cv2
import opencv
from hw1 import load_data


def consecutive_frames_linkage(basis, features, VIDEO_FILE):
    coefficients = basis.T @ features

    x1 = coefficients[0]
    x2 = coefficients[1]

    pairs_x_consecutive = [[x1[i], x1[i + 1]]
                           for i in range(len(x1) - 1)]

    pairs_y_consecutive = [[x2[i], x2[i + 1]]
                           for i in range(len(x2) - 1)]

    v = cv2.VideoCapture(VIDEO_FILE)

    for idx_frame in range(len(x1)):
        clear_output(wait=True)

        fig, axs = plt.subplots(1, 2, figsize=(15, 12))

        axs[0].title.set_text('Distribution of full dataset of dimension {0} on rank = 2 subspace'.format(features.shape[0]))
        axs[0].plot(pairs_x_consecutive, pairs_y_consecutive, 'b-')
        axs[0].plot(x1, x2, 'r*')
        axs[0].set_xlabel('x1')
        axs[0].set_xlabel('x2')

        fig.suptitle('frame {0}'.format(idx_frame))

        axs[0].plot(x1[idx_frame], x2[idx_frame], 'go')
        v.set(cv2.CAP_PROP_POS_FRAMES, idx_frame)
        _, frame = v.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        axs[1].imshow(frame)
        plt.show()

    v.release()
    cv2.destroyAllWindows()
    return True


def consecutive_frames_distance(features):
    d = np.zeros((features.shape[1] - 1, 1))

    for i in range(len(d)):
        d[i, 0] = np.linalg.norm(features[:, i] - features[:, i + 1])

    d_mean = np.mean(d)
    d_std = np.std(d)
    threshold1 = d_mean + 2 * d_std
    threshold2 = d_mean - 2 * d_std

    idx_transitions = np.argwhere(d > threshold1)[:, 0]
    idx_transitions = sorted(np.append(idx_transitions, np.argwhere(d < threshold2)[:, 0]))

    segment_frames = np.array([0])

    for j in range(len(idx_transitions)):
        if j == 0:
            if idx_transitions[j] != 0:
                segment_frames = np.append(segment_frames, [idx_transitions[j], idx_transitions[j] + 1])
            else:
                segment_frames = np.append(segment_frames, 1)

        else:
            if idx_transitions[j] - idx_transitions[j - 1] != 1:
                segment_frames = np.append(segment_frames, [idx_transitions[j], idx_transitions[j] + 1])

    segment_frames = np.append(segment_frames, d.shape[0])

    plt.figure(figsize=(15, 12))
    plt.plot(d)
    plt.plot(np.arange(0, d.shape[0]), np.ones((d.shape[0], 1)) * threshold1, 'g-', label='miu + 3*std')
    plt.plot(np.arange(0, d.shape[0]), np.ones((d.shape[0], 1)) * threshold2, 'r-', label='miu - 3*std')
    plt.plot(idx_transitions, d[idx_transitions], 'r*', label='transitions')
    plt.legend(loc='upper right')
    plt.title('Distance between consecutive embeddings')
    plt.show()

    return d, segment_frames


def sigularvalues(sigma, plotting=False, threshold_representativity=0.9):
    R = sigma.shape[0]

    cumsum_sigma1 = [np.sum(sigma[k:R])
                     for k in range(R)]

    cumsum_sigma2 = [np.sum(sigma[:k])
                     for k in range(R)]

    normalized1 = cumsum_sigma1 / np.sum(sigma)
    normalized2 = cumsum_sigma2 / np.sum(sigma)

    sigma_min_repr1 = np.argwhere(normalized1 <= 1 - threshold_representativity)[0]

    sigma_min_repr2 = np.argwhere(normalized2 >= threshold_representativity)[0]

    if plotting:
        plt.figure()
        plt.title('singular values')
        plt.plot(sigma)
        plt.grid()

        plt.figure()
        plt.title('log singular values')
        plt.plot(np.log(sigma))
        plt.grid()

        plt.figure()
        plt.plot(normalized1, 'b')

        plt.arrow(sigma_min_repr1, 1 - threshold_representativity, 50, 0.1, width=0.01, color='black')
        plt.annotate('{0:.3g}'.format(normalized1[sigma_min_repr1][0]),
                     (sigma_min_repr1 + 55, 1 - threshold_representativity + 0.15),
                     textcoords="offset points",
                     xytext=(5, 5),
                     ha='center')

        plt.plot(normalized2, 'r')

        plt.arrow(sigma_min_repr2, threshold_representativity, 50, -0.1, width=0.01, color='black')
        plt.annotate('{0:.3g}'.format(normalized2[sigma_min_repr2][0]),
                     (sigma_min_repr2 + 55, threshold_representativity - 0.11),
                     textcoords="offset points",
                     xytext=(5, 5),
                     ha='center')

        plt.xlabel('Singular values')
        plt.ylabel('Representativity')

        plt.axvline(x=sigma_min_repr2, color='red', linestyle='dotted', linewidth=2)

        plt.grid()

        plt.show()

    print('We can get {0:.5g}% representativity with {1} singular values.'.format(normalized2[sigma_min_repr2][0] * 100, sigma_min_repr2[0] + 1))

    return sigma_min_repr2 + 1


# Skeletons

def draw_pose(img, skeletons_x, skeletons_y, display_missing_data=False):
    # FIXME set these as constant in setup file
    _PAIRS = [
        (0, 1), (0, 14), (0, 15), (1, 2), (1, 5), (1, 8), (1, 11),  # torso
        (2, 3), (3, 4),  # right arm
        (5, 6), (6, 7),  # left arm
        (8, 9), (9, 10),  # right leg
        (11, 12), (12, 13),  # lef leg
        (14, 16), (15, 17),  # face
    ]

    _DRAW_COLOUR = (255, 0, 0)
    _DRAW_CIRCLE_RADIOUS = 3
    _DRAW_HAND_RADIOUS = 5

    _DRAW_LINE_WIDTH = 2
    _DRAW_COLOUR_HANDS = (0, 0, 255)

    img_height = img.shape[0]
    img_width = img.shape[1]

    valid = (skeletons_x + skeletons_y).astype(bool)  # PODE SE TIRAR SE QUISEREM MESMO VER A MISSING DATA

    for kp1, kp2 in _PAIRS:

        # PODE SE TIRAR ESTE IF SE QUISEREM VER A MISSING DATA - I think it's fixed with the OR, couldn't test tho
        if display_missing_data or valid[kp1] & valid[kp2]:
            x1 = int(skeletons_x[kp1, :] * img_width)
            y1 = int(skeletons_y[kp1, :] * img_height)
            x2 = int(skeletons_x[kp2, :] * img_width)
            y2 = int(skeletons_y[kp2, :] * img_height)

            opencv.line(img, (x1, y1), (x2, y2), _DRAW_COLOUR, _DRAW_LINE_WIDTH)
            # cv2.line(img, (x1, y1), (x2, y2), _DRAW_COLOUR, _DRAW_LINE_WIDTH)

            # if kp2 == 4 or kp2 == 7:
            #    cv2.circle(img, (x1, y1), _DRAW_CIRCLE_RADIOUS, _DRAW_HAND_RADIOUS, -1)
            # else:

            opencv.circle(img, (x1, y1), _DRAW_CIRCLE_RADIOUS, _DRAW_COLOUR, -1)
            opencv.circle(img, (x2, y2), _DRAW_CIRCLE_RADIOUS, _DRAW_COLOUR, -1)
            # cv2.circle(img, (x1, y1), _DRAW_CIRCLE_RADIOUS, _DRAW_COLOUR, -1)
            # cv2.circle(img, (x2, y2), _DRAW_CIRCLE_RADIOUS, _DRAW_COLOUR, -1)

    return True


# General plots


def heatmap_example():
    data = np.random.rand(10, 10)
    print("Our dataset is : ", data)

    plt.figure(figsize=(10, 10))
    heat_map = sns.heatmap(data, linewidth=1, annot=True)
    plt.title("HeatMap using Seaborn Method")
    plt.show()
    return True


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
    return True


def pie_plot(data, title):
    '''
    Plots a pie plot #wow #surprise
    data in the format (..., ('label', value), ...)
    '''
    labels = list(map(lambda x: x[0], data))
    values = list(map(lambda x: x[1], data))
    plt.title(title)
    plt.pie(values, labels=labels, autopct='%1.0f%%')
    plt.tight_layout()
    plt.show()


def skel_stats(data):
    n_complete_skels = np.sum(np.count_nonzero(data, axis=0) < 55)
    n_incomplete_skels = np.shape(data)[1] - n_complete_skels
    pie_plot((('Complete skels', n_complete_skels), ('Incomplete skels', n_incomplete_skels)), 'Skeleton Evaluation')

    n_missing_points = sum((54 - np.count_nonzero(data[1:, :], axis=0)) / 3)
    n_known_points = (data.size - np.shape(data)[1]) - n_missing_points
    pie_plot((('Missing points', n_missing_points), ('Known poins', n_known_points)), 'Point Evaluation')


def skel_info_deriv(data):
    last = int(data[0][-1])
    frames = [i for i in range(last)]
    hist = [0] * (last + 1)
    for i in data[0]:
        hist[int(i)] += 1
    hist = np.ediff1d(np.array(hist))
    # bar_plot(frames, hist, 'Change in Skels\' #', 'Frame', '# of Skels')
    plt.plot(hist)
    plt.show()

if __name__ == '__main__':
    skeli, _, _, _ = load_data('all')

    skel_stats(skeli)
