from curses import beep
from pprint import pprint
from beepy import beep
import matplotlib.pyplot as plt
import sklearn.cluster as skc  # KMeans
import sklearn.manifold as skm  # TSNE
import sklearn.metrics as skmt  # precision, recall, ...
import numpy as np
import matplotlib.cm as cm
import linear_subspaces
from sklearn.manifold import TSNE
from hw1 import load_data, SVD
from operator import itemgetter


def kmeans_clustering(features, rank, n_clusters=4, max_iter=1000):
    # o prof faz esta linha de código
    # fcentered=features- (np.mean(features.T).T * np.ones((1,features.shape[1])))
    fcentered = features - np.mean(features.T).T

    basis_centered, sigma_centered, vT_centered = linear_subspaces.factorisation(fcentered, rank)

    feat_coef = basis_centered.T @ fcentered

    X = feat_coef.T

    kmeans = skc.KMeans(n_clusters=n_clusters, random_state=0, max_iter=max_iter).fit(X)

    idx_features = skc.KMeans(n_clusters=n_clusters, random_state=0, max_iter=max_iter).fit_predict(X)

    return X, kmeans.cluster_centers_, kmeans.labels_, idx_features


# NUMBER OF CLUSTERS TO BE CALCULATED AND NOT GUESSED
# max_iter TO BE EVALUATED. WHAT'S THE BEST? DEPENDS ON DATA? DEPENDS ON # OF CLUSTERS?

# Number of clusters analysis


def clusters_analysis(data, rank):
    def subsample(coefs, size):
        random_indices = np.random.choice(coefs.shape[0], size=size, replace=False)
        return coefs[random_indices, :]

    feat_fact_r = SVD(data, rank)
    feat_coef_r = feat_fact_r[0].T @ data
    feat_coef_r_small = subsample(feat_coef_r.T, 2000)

    C = range(1, rank)
    metric_f = []
    for c in C:
        points, centers, labels, _ = kmeans_clustering(feat_coef_r_small, rank, n_clusters=c, max_iter=10000)
        metric = np.sum([np.linalg.norm(i - centers[n]) for i, n in zip(points, labels)])  # WSS
        metric_f += [metric]
    plt.plot(C, metric_f, 'kx--')
    plt.title('Clustering Study')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Within Sum of Squares')
    plt.show()


def scatter_clusters(points, centers, labels):
    '''Plotting data with colours indicating cluster'''
    plt.title('Distribution of clustered images on 2 dimensions')
    plt.xlabel('x1')
    plt.ylabel('x2')
    colours = cm.rainbow(np.linspace(0, 1, len(centers)))
    for point, label in zip(points, labels):
        plt.scatter(point[0], point[1], color=colours[label], marker='.')
    for i, center in enumerate(centers):
        plt.scatter(center[0], center[1], color='k', marker='+')
    plt.show()


def tsne(dataset, n_components=2):
    """ Used to visualise high dimensional data. To be used after applying k-means
    clustering to high dimensional (centered) data."""
    tsne_features = TSNE(n_components=n_components, init='pca', random_state=0).fit_transform(dataset)

    # dataset = _centre_data()
    # dataset = _reduce_data_dimension()
    # tsne_centered = skm.TSNE(n_components).fit_transform(np.append(skeletons_x_centered, skeletons_y_centered, axis = 0).T)
    # return tsne_centered[:, 0], tsne_centered[:, 1], labels?
    # raise NotImplementedError
    return tsne_features


def point_in_segment(point, segments, segment_ix):
    return point in segments[segment_ix]


def supervised_clustering(dataset, tails, max_iter=1000):
    def _form_segments(tails):

        num_segments = len(tails['begin'])
        groups = [np.arange(tails['begin'][ix], tails['end'][ix] + 1, dtype=int) for ix in range(num_segments)]

        selected_frames = np.hstack(groups)
        return num_segments, groups, selected_frames

    def _expected_labels(groups):
        expected_labels = []
        for cluster_ix in range(len(groups)):
            for _ in range(len(groups[cluster_ix])):
                expected_labels += [cluster_ix]
        return np.array(expected_labels)

    def _correct_labels(predicted_labels, num_segments):
        occurrences = []
        for cluster_ix in range(num_segments):
            occurrence = np.argwhere(predicted_labels == cluster_ix)
            occurrence = occurrence.reshape(1, -1)[0].tolist()
            occurrences += [occurrence]

        occurrences = sorted(occurrences, key=itemgetter(0))

        for cluster_ix in range(num_segments):
            for frame in occurrences[cluster_ix]:
                predicted_labels[frame] = cluster_ix

        return predicted_labels

    num_segments, groups, selected_frames = _form_segments(tails)
    expected_labels = _expected_labels(groups)
    dataset = dataset[:, selected_frames]
    *_, predicted_labels, _ = kmeans_clustering(dataset, num_segments, n_clusters=num_segments, max_iter=max_iter)
    predicted_labels = _correct_labels(predicted_labels, num_segments)

    return num_segments, expected_labels, predicted_labels


if __name__ == '__main__':
    import setup

    # coefs,centres, labels, _ = kmeans_clustering(dev, 100)
    # scatter_clusters(coefs, centres, labels)

    # Selected these frames by analysing the video
    # 5895 - 5906
    # 1994 - 2005
    # 3337 - 3345
    # 700 - 711
    # 200 - 222
    # 121 - 134
    # 51 - 96

    tails = {'begin': [5895, 1994, 3337, 700, 200, 121, 51], 'end': [5906, 2005, 3345, 711, 222, 134, 96]}
    num_segments, expected_labels, predicted_labels = supervised_clustering(setup.FEATURES, tails)

    print(predicted_labels)

    print(skmt.classification_report(expected_labels, predicted_labels, labels=np.arange(num_segments)))

    dataset = 'all'
    _, _, feat, _ = load_data(dataset)

    # feat_fact = SVD(feat)
    feat_fact_r8 = SVD(feat, 8)
    feat_coef_r8 = feat_fact_r8[0].T @ feat

    kmeans = kmeans_clustering()
    # feat_coef_r2 = feat_fact_r2[0].T @ feat
    # feat_coef_r2 = np.diag(feat_fact_r2[1]) @ feat_fact_r2[2]

    # connected_scatter_plot(feat_coef_r2, f'Distribution of full dataset of dimension {feat.shape[0]} on rank = 2 subspace', 'x1', 'x2')

    # Treating coefficients and clustering

    # X = np.array(list(zip(feat_coef_r2[0], feat_coef_r2[1])))
    # kmeans = skc.KMeans(n_clusters=4, random_state=0, max_iter=1000).fit(X)

    # coefs, centres, labels = clustering(feat, dimension=2)
    # scatter_clusters(coefs, centres, labels)

    # scatter_clusters(X, kmeans.cluster_centers_, kmeans.labels_)
