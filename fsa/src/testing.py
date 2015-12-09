#!/usr/bin/env python
# -*- coding: utf-8 -*-
# *************************************** #
#
#  Author:  Semen Budenkov
#  Date:    01/10/2015
#
# *************************************** #

from __future__ import print_function, division
import csv
import json
from collections import Counter
import cPickle as pickle
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from skfuzzy import cluster

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import rc, markers
import matplotlib.colors as clrs
import matplotlib.cm as cmx
import numpy as np
from scipy import stats
from visualization import find_boundary
from sklearn import svm
from sklearn.covariance import EllipticEnvelope


# from train_model import preprocess
import time
from preprocessing import preprocess
from fuzzy_clusters import get_clusters


def test(categories):
    use_stem = True
    test_data = []
    test_labels = []
    tsv_out1 = open(".\\src\\test.tsv", "wb")
    tsv_out = csv.writer(tsv_out1, delimiter='\t')

    test_json = open(".\\src\\test.json")
    count_all = Counter()
    for r in test_json:
        tweet = json.loads(r)
        if (tweet["lang"] != "ru"):
            continue
        # Create a list with all the terms
        terms_all = [term for term in preprocess(tweet['text'], True)]
        # Update the counter
        count_all.update(terms_all)
        # tokens = preprocess(tweet['text'], True)
        # for token in tokens:
        #     print token

        # print tweet["text"].encode(sys.stdout.encoding, errors='replace')
        # tsv_out.writerow(["hz", tweet["text"].encode("utf-8")])
    for token in count_all.most_common(5):
        print(token[0] + ":" + str(token[1]))
    exit()
    tsv_out1.close()
    # exit(0)

    # test_in = open(".\\data\\parsed\\ttk_train.tsv")
    test_in = open(".\\src\\test.tsv")
    test_in = csv.reader(test_in, delimiter='\t')

    fin1 = open('vectorizer.pk', 'r')
    vectorizer = pickle.load(fin1)

    fin2 = open('classifier_linear.pk', 'r')
    classifier_linear = pickle.load(fin2)

    test_data, test_labels = preprocess(test_in, use_stem)
    test_vectors = vectorizer.transform(test_data)

    prediction_linear = classifier_linear.predict(test_vectors)

    print("Results for SVC(kernel=linear)")
    print(classification_report(test_labels, prediction_linear))
    with open("result_linear_test.txt", "wb") as result_out:
        i = 0
        for s in prediction_linear:
            if (test_labels[i] != prediction_linear[i]):
                result_out.write(
                    test_labels[i] + " : " + prediction_linear[i] + '\t' + test_data[i].encode("utf-8") + '\n')
            i += 1


if __name__ == '__main__':
    rc('font', family='Arial')
    # rc('font', family='Ubuntu')

    soft = False
    outliers_fraction = 0.3

    print("INFO: Test word2vec model")
    # model = Word2Vec.load("../models/300features_20minwords_10context_full")
    model = Word2Vec.load("../models/300features_40minwords_10context_full")
    # model = Word2Vec.load_word2vec_format('../models/news.model.bin.gz', binary=True)
    # model = Word2Vec.load_word2vec_format('../models/ruscorpora.model.bin.gz', binary=True)
    word_vectors = model.syn0

    print('INFO: Shape word_vectors: ', word_vectors.shape)
    n_components = 2
    pca = PCA(n_components)
    pca_vectors = pca.fit_transform(word_vectors)

    tsne = TSNE(n_components=n_components, learning_rate=0)
    tsne_vectors = tsne.fit_transform(np.asfarray(word_vectors, dtype='float64'))

    num_clusters = 2
    start = time.time()  # Start time

    print('INFO: Clustering: ', num_clusters, ' clusters')
    # if not soft:
    kmeans_clustering = KMeans(n_clusters=num_clusters)
    kclusters = kmeans_clustering.fit_predict(word_vectors)
    # else:
    word_vectors_transpose = word_vectors.transpose()
    cntr, u, u0, d, jm, p, fpc = cluster.cmeans(word_vectors_transpose,
                                                num_clusters,
                                                2,
                                                error=1e-4,
                                                maxiter=300,
                                                init=None)
    # cclusters = np.argmax(u, axis=0)
    # print(cclusters, cclusters.shape)
    # cclusters_fuzzy = get_clusters(u, limit=1/num_clusters)
    cclusters_fuzzy = get_clusters(u, limit=0.35)
    # cl = get_clusters(u, n_components=1)
    # print(cclusters_fuzzy[-4], cclusters_fuzzy.shape)
    # exit(0)

    end = time.time()
    elapsed = end - start
    print("INFO: Time of clustering: ", elapsed, "seconds")

    jet = cm = plt.get_cmap('jet')
    cNorm = clrs.Normalize(vmin=0, vmax=num_clusters)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

    for X_transformed, title in [(pca_vectors, "PCA"), (tsne_vectors, "TSNE")]:
        if title == "TSNE": break
        fig = plt.figure()
        fig.canvas.set_window_title(title)
        # ax = fig.add_subplot(111, projection='3d')
        # plt.figure(figsize=(8, 8))
        # plt.scatter(X_transformed[:, 0],
        #             X_transformed[:, 1])
        # print X_transformed.shape
        print("INFO: Dimensionality reduction method: ", title)

        for i, target_name in zip(range(num_clusters), [str(ix) + ' кластер' for ix in range(num_clusters)]):
            cluster_color = scalarMap.to_rgba(i)
            m = markers.MarkerStyle.filled_markers[i]
            plt.subplot(211)
            plt.title('K-means: ' + str(num_clusters) + ' clusters')
            # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
            plt.scatter(X_transformed[kclusters == i, 0],
                        X_transformed[kclusters == i, 1],
                        marker=m,
                        c=cluster_color
                        # ,
                        # label=target_name.decode('utf8')
                        )

            x, y = find_boundary(X_transformed[kclusters == i, 0],
                                 X_transformed[kclusters == i, 1], 5)
            plt.plot(x, y, '-k', lw=2., color=cluster_color)

            # create a mesh to plot in
            h = .02  # step size in the mesh
            x_min, x_max = X_transformed[kclusters == i, 0].min() - 1, X_transformed[kclusters == i, 0].max() + 1
            y_min, y_max = X_transformed[kclusters == i, 1].min() - 1, X_transformed[kclusters == i, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))

            clf = EllipticEnvelope(contamination=.1)
            clf.fit(X_transformed[kclusters == i])

            pred = clf.decision_function(X_transformed[kclusters == i]).ravel()
            threshold = stats.scoreatpercentile(pred,
                                                100 * outliers_fraction)
            print("INFO: Cluster: ", i, " Threshold: ", threshold)

            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])

            Z = Z.reshape(xx.shape)
            # plt.contour(xx, yy, Z,
            #             levels=[threshold],
            #             linewidths=2,
            #             linestyles='solid',
            #             colors=(cluster_color,))

            # plt.contourf(xx, yy, Z, levels=[threshold, Z.max()],
            #              colors='orange')
            # plt.contourf(xx, yy, Z, levels=[threshold, Z.max()],
            #              colors=cluster_color)
            # for label, x, y in zip(model.index2word, X_transformed[:, 0], X_transformed[:, 1]):
            #     plt.annotate(
            #         label,
            #         (x, y),
            #         xytext=(0, 3),
            #         textcoords = 'offset points', ha = 'left', va = 'bottom'
            #         # ,
            #         # family="fantasy"
            #     )

            plt.subplot(212)
            plt.title('C-means: ' + str(num_clusters) + ' clusters')

            cluster_mask = []
            for cluster_row in cclusters_fuzzy:
                if i in cluster_row:
                    cluster_mask.append(True)
                else:
                    cluster_mask.append(False)

            cluster_mask = np.asarray(cluster_mask)
            # print(cluster_mask, cluster_mask.shape)

            plt.scatter(
                        # X_transformed[cclusters == i, 0],
                        # X_transformed[cclusters == i, 1],
                        X_transformed[cluster_mask, 0],
                        X_transformed[cluster_mask, 1],
                        marker=m,
                        c=cluster_color
                        # ,
                        # label=target_name.decode('utf8')
                        )
            x, y = find_boundary(X_transformed[cluster_mask, 0],
                                 X_transformed[cluster_mask, 1], 5)
            plt.plot(x, y, '-k', lw=2., color=cluster_color)

            # for label, x, y in zip(model.index2word, X_transformed[:, 0], X_transformed[:, 1]):
            #     plt.annotate(
            #         label,
            #         (x, y),
            #         xytext=(0, 3),
            #         textcoords = 'offset points', ha = 'left', va = 'bottom'
            #     )

            # plt.setp(plt.get_xticklabels(), visible=False)
            # if "Incremental" in title:
            #     err = np.abs(np.abs(X_pca) - np.abs(X_ipca)).mean()
            #     plt.title(title + " of iris dataset\nMean absolute unsigned error "
            #               "%.6f" % err)
            # else:
            #     plt.title(title + " of iris dataset")
            # plt.legend(loc="best")
            # plt.axis([-4, 4, -1.5, 1.5])

    plt.show()
