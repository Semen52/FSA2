#!/usr/bin/env python
# -*- coding: utf-8 -*-
# *************************************** #
#
#  Author:  Semen Budenkov
#  Date:    01/10/2015
#
# *************************************** #

import os
import time
import numpy as np

from gensim.models      import Word2Vec
from sklearn.cluster    import MiniBatchKMeans as kmeans
from sklearn.cluster    import KMeans
from sklearn.manifold   import TSNE
from sklearn.decomposition import PCA
from sklearn import     manifold
from sklearn.utils import check_random_state

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from matplotlib import rc
import matplotlib.colors as clrs
import matplotlib.cm as cmx


def load_word2vec(dir):
    # new: since github has a 100M limit, load from a bunch of files in
    # a directory
    word2vec = {}
    for path in os.listdir(dir):
        iword2vec = {}
        # load the word2vec features.
        with open(os.path.join(dir, path), 'r') as fin:
            if path == 'vectors0.txt':
                next(fin)  # skip information on first line
            for line in fin:
                items = line.replace('\r', '').replace('\n', '').split(' ')
                if len(items) < 10: continue
                word = items[0]
                vect = np.array([float(i) for i in items[1:] if len(i) > 1])
                iword2vec[word] = vect

        word2vec.update(iword2vec)

    return word2vec


def get_furthest_word(words, word2vect):
    vectlist = []
    for word in words:
        # unknown word?
        if word not in word2vect: return word
        # normalize.
        vectlist.append(word2vect[word] / np.linalg.norm(word2vect[word]))
    mean = np.array(vectlist).mean(axis=0)
    mean = mean / np.linalg.norm(mean)

    # figure out which is furthest
    dists = [np.linalg.norm(v - mean) for v in vectlist]
    return words[np.argmax(dists)]


def cluster_vects(word2vect):
    # use sklearn minibatch kmeans to cluster the vectors.
    clusters = kmeans(n_clusters=25, max_iter=10, batch_size=200,
                      n_init=1, init_size=2000)
    X = np.array([i.T for i in word2vect.itervalues()])
    y = [i for i in word2vect.iterkeys()]

    print 'fitting kmeans, may take some time'
    clusters.fit(X)
    print 'done.'

    # now we can get a mapping from word->label
    # which will let us figure out which other words are in the same cluster
    return {word: label for word, label in zip(y, clusters.labels_)}


def words_in_cluster(word, word_to_label):
    # sorry, this is O(n), n is pretty large
    # it could be O(k), k=cluster size, but that would cost more memory
    label = word_to_label[word]
    # get the other words with this label
    similar_words = [key for key, val in word_to_label.iteritems() if val == label]
    return similar_words

def show_clusters():
    print("Show clusters on 2D")
    # model = Word2Vec.load("..\\models\\300features_40minwords_10context_mystem")
    model = Word2Vec.load_word2vec_format('..\\models\\news.model.bin.gz', binary=True)

    # Set "c" (num_clusters) to be 1/5th of the vocabulary size, or an
    # average of 5 words per cluster
    word_vectors = model.syn0
    # word_vectors = word_vectors.transpose()

    num_clusters = 5

    start = time.time()  # Start time

    kmeans_clustering = KMeans(n_clusters=num_clusters)
    idx = kmeans_clustering.fit_predict(word_vectors)

    end = time.time()
    elapsed = end - start
    print "Time taken for clustering: ", elapsed, "seconds."

    word_centroid_map = dict(zip(model.index2word, idx))

    print('Shape index2word: ', len(model.index2word))
    # print model.index2word[0], idx[0]
    # exit(0)

    # Print the first ten clusters
    # for cluster in xrange(0, 2):
    #     # Print the cluster number
    #     print "\nCluster %d" % cluster
    #
    #     # Find all of the words for that cluster number, and print them out
    #     words = []
    #     for i in xrange(0, len(word_centroid_map.values())):
    #         if word_centroid_map.values()[i] == cluster:
    #             words.append(word_centroid_map.keys()[i])
    #             print word_centroid_map.keys()[i]
    #     # print words

    print('Shape word_vectors: ', word_vectors.shape)
    n_components = 2
    pca = PCA(n_components)
    pca_vectors = pca.fit_transform(word_vectors)

    print('Shape pca_vectors: ', pca_vectors.shape)
    jet = cm = plt.get_cmap('jet')
    cNorm  = clrs.Normalize(vmin=0, vmax=num_clusters)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

    for X_transformed, title in [(pca_vectors, "PCA")]:
        plt.figure(figsize=(8, 8))
        for i, target_name in zip(range(num_clusters), [str(ix) + ' кластер' for ix in range(num_clusters)]):
            colorVal = scalarMap.to_rgba(i)
            plt.scatter(X_transformed[idx == i, 0],
                        X_transformed[idx == i, 1],
                        c=colorVal,
                        label=target_name.decode('utf8'))

        for label, x, y in zip(model.index2word, X_transformed[:, 0], X_transformed[:, 1]):
            plt.annotate(
                label,
                (x, y),
                # xy=(x, y),
                xytext=(0, 3),
                # ,
                textcoords = 'offset points', ha = 'left', va = 'bottom'
                # ,
                # bbox = dict(boxstyle='round,pad=0.5', fc = 'yellow', alpha = 0.5),
                # arrowprops = dict(arrowstyle='->', connectionstyle='arc3,rad=0')
            )

        # plt.setp(plt.get_xticklabels(), visible=False)
        # if "Incremental" in title:
        #     err = np.abs(np.abs(X_pca) - np.abs(X_ipca)).mean()
        #     plt.title(title + " of iris dataset\nMean absolute unsigned error "
        #               "%.6f" % err)
        # else:
        #     plt.title(title + " of iris dataset")
        plt.legend(loc="best")
        # plt.axis([-4, 4, -1.5, 1.5])

    plt.show()

    # for i in xrange(0, 10):
    #     print model.index2word[i]
        # print word_vectors[i]
    exit(0)

    # X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    # model = TSNE(n_components=2, random_state=0)
    # model.fit_transform(X)
    # Next line to silence pyflakes.
    Axes3D

    # Variables for manifold learning.
    n_neighbors = 10
    n_samples = 1000

    # Create our sphere.
    random_state = check_random_state(0)
    p = random_state.rand(n_samples) * (2 * np.pi - 0.55)
    t = random_state.rand(n_samples) * np.pi

    # Sever the poles from the sphere.
    indices = ((t < (np.pi - (np.pi / 8))) & (t > ((np.pi / 8))))
    colors = p[indices]
    x, y, z = np.sin(t[indices]) * np.cos(p[indices]), \
        np.sin(t[indices]) * np.sin(p[indices]), \
        np.cos(t[indices])

    # Plot our dataset.
    fig = plt.figure(figsize=(15, 8))
    plt.suptitle("Manifold Learning with %i points, %i neighbors"
                 % (1000, n_neighbors), fontsize=14)

    ax = fig.add_subplot(251, projection='3d')
    ax.scatter(x, y, z, c=p[indices], cmap=plt.cm.rainbow)
    try:
        # compatibility matplotlib < 1.0
        ax.view_init(40, -10)
    except:
        pass

    sphere_data = np.array([x, y, z]).T

    # Perform Locally Linear Embedding Manifold learning
    methods = ['standard', 'ltsa', 'hessian', 'modified']
    labels = ['LLE', 'LTSA', 'Hessian LLE', 'Modified LLE']

    for i, method in enumerate(methods):
        t0 = time()
        trans_data = manifold\
            .LocallyLinearEmbedding(n_neighbors, 2,
                                    method=method).fit_transform(sphere_data).T
        t1 = time()
        print("%s: %.2g sec" % (methods[i], t1 - t0))

        ax = fig.add_subplot(252 + i)
        plt.scatter(trans_data[0], trans_data[1], c=colors, cmap=plt.cm.rainbow)
        plt.title("%s (%.2g sec)" % (labels[i], t1 - t0))
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')

    # Perform Isomap Manifold learning.
    t0 = time()
    trans_data = manifold.Isomap(n_neighbors, n_components=2)\
        .fit_transform(sphere_data).T
    t1 = time()
    print("%s: %.2g sec" % ('ISO', t1 - t0))

    ax = fig.add_subplot(257)
    plt.scatter(trans_data[0], trans_data[1],  c=colors, cmap=plt.cm.rainbow)
    plt.title("%s (%.2g sec)" % ('Isomap', t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

    # Perform Multi-dimensional scaling.
    t0 = time()
    mds = manifold.MDS(2, max_iter=100, n_init=1)
    trans_data = mds.fit_transform(sphere_data).T
    t1 = time()
    print("MDS: %.2g sec" % (t1 - t0))

    ax = fig.add_subplot(258)
    plt.scatter(trans_data[0], trans_data[1],  c=colors, cmap=plt.cm.rainbow)
    plt.title("MDS (%.2g sec)" % (t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

    # Perform Spectral Embedding.
    t0 = time()
    se = manifold.SpectralEmbedding(n_components=2,
                                    n_neighbors=n_neighbors)
    trans_data = se.fit_transform(sphere_data).T
    t1 = time()
    print("Spectral Embedding: %.2g sec" % (t1 - t0))

    ax = fig.add_subplot(259)
    plt.scatter(trans_data[0], trans_data[1],  c=colors, cmap=plt.cm.rainbow)
    plt.title("Spectral Embedding (%.2g sec)" % (t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

    # Perform t-distributed stochastic neighbor embedding.
    t0 = time()
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    trans_data = tsne.fit_transform(sphere_data).T
    t1 = time()
    print("t-SNE: %.2g sec" % (t1 - t0))

    ax = fig.add_subplot(250)
    plt.scatter(trans_data[0], trans_data[1],  c=colors, cmap=plt.cm.rainbow)
    plt.title("t-SNE (%.2g sec)" % (t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

    plt.show()

###################################
rc('font', family='Arial')

show_clusters()
