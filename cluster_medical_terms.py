import sys
sys.path.insert(0,'/vol/medic02/users/ag6516/miniconda/envs/PyTorch27/lib/python2.7/site-packages')
sys.path.insert(0,'/vol/medic02/users/ag6516/miniconda/envs/PyTorch27/lib/python2.7/site-packages/setuptools-27.2.0-py2.7.egg')
sys.path.insert(0,'/vol/medic02/users/ag6516/miniconda/envs/PyTorch27/lib/python2.7/site-packages/IPython/extensions')
sys.path

# from gensim.models import word2vec
#
# def w2v(s1,s2,wordmodel):
#     if s1==s2:
#         return 1.0
#
#     s1words=s1.split()
#     s2words=s2.split()
#     s1wordsset=set(s1words)
#     s2wordsset=set(s2words)
#     vocab = wordmodel.vocab #the vocabulary considered in the word embeddings
#     if len(s1wordsset & s2wordsset)==0:
#         return 0.0
#     for word in s1wordsset.copy(): #remove sentence words not found in the vocab
#         if (word not in vocab):
#             s1words.remove(word)
#     for word in s2wordsset.copy(): #idem
#         if (word not in vocab):
#             s2words.remove(word)
#     return wordmodel.n_similarity(s1words, s2words)
#
# if __name__ == '__main__':
#     wordmodelfile="/home/maali/GoogleNews-vectors-negative300.bin.gz"
#     wordmodel= word2vec.Word2Vec.load_word2vec_format(wordmodelfile, binary=True)
#     s1="As California Bounces Back , Governor Calls For Lofty Goals"
#     s2="With California Rebounding, Governor Pushes Big Projects"
#     print "sim(s1,s2) = ", w2v(s1,s2,wordmodel),"/1."
#     s3="Special measures for Beijing polution"
#     s4="Smog cloud blankets Beijing"
#     print "sim(s3,s4) = ", w2v(s3,s4,wordmodel),"/1."


import collections
import nltk
nltk.data.path.append("/vol/medic02/users/ag6516/nltk_data");
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
import csv

from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn import datasets
from numpy import zeros
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist, pdist
from scipy.cluster.hierarchy import ward, dendrogram
from sklearn.metrics.pairwise import cosine_similarity
import random
from gensim import corpora, models, similarities
#import seaborn as sns
import pandas as pd

def compute_improvement(ref, new):
    return float(ref-new)/ref

def ikmeans(km, X, min_improvement=.01):
    ''' incremental kmeans; split worst cluster based on the silhouette score '''
    K = km.n_clusters

    labels = km.labels_
    # compute label distribution
    labels_ratio = np.histogram(labels,bins=len(set(labels)))[0]
    labels_ratio = np.array(labels_ratio, dtype=float)/len(labels)
    scores = metrics.silhouette_samples(X, labels, metric='euclidean')
    score = scores.mean()

    # measure global performance of each cluster
    k_score = zeros(K)
    for k in range(K):
        idx = np.where(labels==k)
        k_score[k] = scores[idx].mean()

    # identify the cluster to split where population higher then min_population_ratio
    idx = np.where(labels_ratio>0.01)[0]
    worst_score = k_score[idx].max()
    worst_idx = np.where(k_score==worst_score)[0]
    if len(worst_idx)>1:
        print "several worst k (%i)" %len(worst_idx)
    worst_k = worst_idx[0]
    print "worst cluster -> %i" %(worst_k)

    # split worst cluster
    idx = np.where(labels==worst_k)
    X_k = X[idx]
    if X_k.shape[1]<=2:
        print "not enough data points to split"
        return
    skm = KMeans(n_clusters=2, random_state=1).fit(X_k)

    # measure improvement with the 2 new clusters
    ikm = KMeans(n_clusters=K+1, random_state=1)
    new_centers = np.array(km.cluster_centers_).tolist()
    new_centers.remove(new_centers[worst_k])
    [new_centers.append(center) for center in skm.cluster_centers_]
    ikm.cluster_centers_ = np.array(new_centers)
    ilabels = ikm.predict(X)
    ikm.labels_ = ilabels
    new_score = metrics.silhouette_score(X, ilabels, metric='euclidean')

    improvement = compute_improvement(score, new_score)

    if improvement > min_improvement:
        print "increase k (%2.2f%%)" %(improvement*100)
        return ikmeans(ikm, X)
    else:
        print "improvement %s->%s = %2.2f%%" %(K, K+1, improvement*100)
        print "best k = %i (score = %.3f)" %(K, new_score)
        return km, new_score

def cluster_ikmeans(sentences, nb_of_clusters=5):
    tfidf_vectorizer = TfidfVectorizer(tokenizer=word_tokenizer,
                                       stop_words=stopwords.words('english'),
                                       max_df=0.999,
                                       min_df=0.001,
                                       lowercase=True)
    #builds a tf-idf matrix for the sentences
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences).toarray()

    km = KMeans(n_clusters=nb_of_clusters, random_state=1).fit(tfidf_matrix)
    best_km, best_score = ikmeans(km, tfidf_matrix)
    km = KMeans(n_clusters=best_km.n_clusters, random_state=1).fit(tfidf_matrix)
    score_ref = metrics.silhouette_score(tfidf_matrix, km.labels_, metric='euclidean')
    print "ref score: %.3f" %(score_ref)
    print "ikmeans improvement %2.2f%%" %(compute_improvement(score_ref, best_score)*100)

def word_tokenizer(text):
    #tokenizes and stems the text
    tokens = word_tokenize(text)
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens if t not in stopwords.words('english')]
    return tokens


def tfidf_vectorise(sentences):
    tfidf_vectorizer = TfidfVectorizer(tokenizer=word_tokenizer,
                                        stop_words=stopwords.words('english'),
                                        max_df=0.99,
                                        min_df=0.01,
                                        lowercase=True)
    #builds a tf-idf matrix for the sentences
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    return tfidf_matrix

def plot_kmeans_pca(sentences, num_clusters, num_samples=100):
    # Visualise clusters
    data = tfidf_vectorise(sentences).toarray()
    kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
    kmeans.fit(data)

    reduced_data = PCA(n_components=2).fit_transform(data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1

    plt.figure(1)
    plt.clf()

    colours = plt.cm.jet(np.linspace(0, 1, num_clusters))

    for i, x, y in zip(range(num_samples), reduced_data[:, 0], reduced_data[:, 1]):
        plt.annotate(sentences[i], xy=(x, y), xytext=(0, 0), textcoords='offset points', color=colours[kmeans.labels_[i]])

    #plt.title('K-means Clustering of Disease Terms from Chest X-Ray Reports (reduced with PCA)\n')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()

def cluster_sentences(sentences, nb_of_clusters=5):

    data = tfidf_vectorise(sentences).toarray()
    kmeans = KMeans(init='k-means++', n_clusters=nb_of_clusters, n_init=10)
    kmeans.fit(data)

    clusters = collections.defaultdict(list)
    for i, label in enumerate(kmeans.labels_):
        clusters[label].append(i)
    return dict(clusters)

def kmeans_silhouette(sentences, k_range):
    tfidf_matrix = tfidf_vectorise(sentences)

    s = []
    # b = []
    # i=0
    for n_clusters in k_range:

        print n_clusters, 'th iteration'
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(tfidf_matrix)

        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        # bic = compute_bic(kmeans, tfidf_matrix.toarray())

        # b.append(bic)
        s.append(silhouette_score(tfidf_matrix, labels, metric='euclidean'))

        #fig, ax1 = plt.subplots()
        #ax1.plot(s, 'b-')
        #ax1.set_xlabel('k')
        # Make the y-axis label, ticks and tick labels match the line color.
        #ax1.set_ylabel('Silhouette', color='b')
        #ax1.tick_params('y', colors='b')

        #ax2 = ax1.twinx()
        #ax2.plot(b, 'r.')
        #ax2.set_ylabel('BIC', color='r')
        #ax2.tick_params('y', colors='r')

        #fig.tight_layout()
        #plt.show()

        plt.plot(k_range, s)
        plt.ylabel("Silouette")
        plt.xlabel("k")
        plt.title("Silouette for K-means cell's behaviour")
        plt.show()
        #sns.despine()

def compute_bic(kmeans,X):
    """
    Computes the BIC metric for a given clusters

    Parameters:
    -----------------------------------------
    kmeans:  List of clustering object from scikit learn

    X     :  multidimension np array of data points

    Returns:
    -----------------------------------------
    BIC value
    """
    # assign centers and labels
    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_
    #number of clusters
    m = kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)
    #size of data set
    N, d = X.shape

    #compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * sum([sum(cdist(X[np.where(labels == i)], [centers[0][i]],
                                                           'euclidean')**2) for i in range(m)])

    const_term = 0.5 * m * np.log(N) * (d+1)

    BIC = np.sum([n[i] * np.log(n[i]) -
                  n[i] * np.log(N) -
                  ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
                  ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term

    return(BIC)

def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

def dendrogram_clusters(sentences):
    # Heirarchical clustering
    dist = 1 - cosine_similarity(tfidf_vectorise(sentences))
    linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances

    fig, ax = plt.subplots(figsize=(15, 20)) # set size
    ax = fancy_dendrogram(linkage_matrix,
                          #truncate_mode='lastp',  # show only the last p merged clusters
                          #p=12,  # show only the last p merged clusters
                          orientation="right",
                          labels=sentences,
                          #leaf_font_size=10,
                          max_d=10);

    clusters = collections.defaultdict(list)
    old_color = ax['color_list'][0]
    cluster = 0
    for i, leaf in enumerate(ax['ivl']):
        if i < len(ax['color_list']):
            new_color = ax['color_list'][i]
        else:
            new_color = 'w'

        if new_color == old_color:
            clusters[cluster].append(leaf)
        else:
            old_color = new_color
            cluster += 1
            clusters[cluster].append(leaf)


    plt.tick_params( \
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')

    #plt.tight_layout() #show plot with tight layout
    plt.show()

    return dict(clusters)

def lda_topic_clusters(sentences, save=False):
    processed_sentences = [word_tokenizer(sentence) for sentence in sentences]

    #create a Gensim dictionary from the texts
    dictionary = corpora.Dictionary(processed_sentences)

    #remove extremes (similar to the min/max df step used when creating the tf-idf matrix)
    dictionary.filter_extremes(no_below=1, no_above=0.8)

    #convert the dictionary to a bag of words corpus for reference
    corpus = [dictionary.doc2bow(sentence) for sentence in processed_sentences]

    lda = models.LdaModel(corpus, num_topics=10,
                          id2word=dictionary,
                          update_every=5,
                          chunksize=10000,
                          passes=100)

    topics = lda.show_topics(formatted=False, num_words=20)

    if save:
        with open('lda_clusters.txt', 'w') as csvf:
            csvw = csv.writer(csvf, delimiter='|')
            for topic in topics:
                csvw.writerow(topic)

    return lda

def lda_perplexity(sentences, t_max=1000, steps=10):
    processed_sentences = [word_tokenizer(sentence) for sentence in sentences]

    #create a Gensim dictionary from the texts
    dictionary = corpora.Dictionary(processed_sentences)

    #remove extremes (similar to the min/max df step used when creating the tf-idf matrix)
    dictionary.filter_extremes(no_below=1, no_above=0.8)

    #convert the dictionary to a bag of words corpus for reference
    corpus = [dictionary.doc2bow(sentence) for sentence in processed_sentences]

    # split into train and test - random sample, but preserving order
    train_size = int(round(len(corpus)*0.8))
    train_index = sorted(random.sample(xrange(len(corpus)), train_size))
    test_index = sorted(set(xrange(len(corpus)))-set(train_index))
    train_corpus = [corpus[i] for i in train_index]
    test_corpus = [corpus[j] for j in test_index]

    number_of_words = sum(cnt for document in test_corpus for _, cnt in document)
    parameter_list = range(5, t_max, steps)
    grid = collections.defaultdict(list)

    for parameter_value in parameter_list:
        print "starting pass for parameter_value = %.3f" % parameter_value
        model = models.LdaMulticore(corpus=train_corpus,
                                    workers=None,
                                    id2word=dictionary,
                                    num_topics=parameter_value,
                                    iterations=50)

        perplex = model.bound(test_corpus) # this is model perplexity not the per word perplexity
        print "Total Perplexity: %s" % perplex
        grid[parameter_value].append(perplex)

        per_word_perplex = np.exp2(-perplex / number_of_words)
        print "Per-word Perplexity: %s" % per_word_perplex
        grid[parameter_value].append(per_word_perplex)
        #model.save(data_path + 'ldaMulticore_i10_T' + str(parameter_value) + '_training_corpus.lda')
        print

    for numtopics in parameter_list:
        print numtopics, '\t',  grid[numtopics]

    df = pd.DataFrame(grid)
    ax = plt.figure(figsize=(7, 4), dpi=300).add_subplot(111)
    df.iloc[1].transpose().plot(ax=ax,  color="#254F09")
    plt.xlim(parameter_list[0], parameter_list[-1])
    plt.ylabel('Perplexity')
    plt.xlabel('topics')
    plt.title('')
    plt.savefig('gensim_multicore_i10_topic_perplexity.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show()


if __name__ == "__main__":
    d_labeld_data = []
    with open('d_f_p_labeled_reports.txt', 'r') as df:
        reader = csv.reader(df, delimiter='|')
        for row in reader:
            report = []
            for i in range(len(row)):
                if i != 0 and row[i] != '':
                    report.append(row[i].replace('_', ' '))
                    #d_labeld_data.append(row[i].replace('_', ' '))
            d_labeld_data.append(', '.join(report))

    #sentences = sorted(set(d_labeld_data))
    sentences = random.sample(sorted(set(d_labeld_data)), len(sorted(set(d_labeld_data))))
    nclusters= 200

    # ----------------- K-Means -------------------
    #clusters = cluster_sentences(sentences, nclusters)
    #plot_kmeans_pca(sentences, nclusters, 100)


    # ----------------- Heirarchical Clustering -------------
    #d_clusters = dendrogram_clusters(sentences)


    # -------------------- LDA -------------------------

    #lda_topic_clusters(sentences)
    #lda_perplexity(sentences)

    #for cluster in range(nclusters):
    #    print "cluster ",cluster,":"
    #    for i, sentence in enumerate(clusters[cluster]):
    #        print "\tsentence ", i, ": ", sentences[sentence]


