def kmeans_clustering(vectors, num_clusters):
    from sklearn.cluster import KMeans
    # docvecs = model.docvecs
    kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
    kmeans.fit(vectors)
    return kmeans.labels_

def plot_pca(vectors, reports, kmeans_labels, num_clusters, num_samples, print_text=True):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    
    reports_tuple = np.array(zip(vectors, reports, kmeans_labels))
    reports_tuple[np.random.choice(len(reports_tuple),num_samples)]
    ss_vectors = reports_tuple[:,0].tolist()
    ss_reports = reports_tuple[:,1].tolist()
    ss_kmeans_labels = reports_tuple[:,2].tolist()
    
    reduced_data = PCA(n_components=2).fit_transform(ss_vectors)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    #h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    x_min, x_max = reduced_data[:, 0].min(), reduced_data[:, 0].max()
    y_min, y_max = reduced_data[:, 1].min(), reduced_data[:, 1].max()

    plt.figure()
    plt.clf()

    colours = plt.cm.jet(np.linspace(0, 1, num_clusters))
    
    for i, x, y in zip(range(num_samples), reduced_data[:, 0], reduced_data[:, 1]):
        if print_text:
            plt.annotate(ss_reports[i], xy=(x, y), xytext=(0, 0), textcoords='offset points', color=colours[ss_kmeans_labels[i]])
        else:
            plt.scatter(x, y, color=colours[ss_kmeans_labels[i]], alpha=.8, lw=2)
                                        
    #plt.title('K-means Clustering of Knee X-ray Reports (reduced with PCA)\n')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()

def save_clusters_to_json(labels, reports, kmeans_labels, num_clusters, filename):
    import json
    clusters = {}
    for c in range(num_clusters):
        cluster = [(i,j) for i,j,k in zip(labels, reports, kmeans_labels) if k==c]
        clusters["cluster{0}".format(c)] = dict(cluster)
    
    with open(filename, 'w') as fp:
        json.dump(clusters, fp)
    
def load_clusters_from_json(filename):
    import json
    with open(filename) as data:
        clusters = json.load(data)
    return clusters
    
def visualise_word_clusters2(clusters, stoplist):
    import nltk
    import numpy as np
    import matplotlib.pyplot as plt
    from nltk.corpus import stopwords
    from wordcloud import WordCloud
    
    for key, cluster in clusters.iteritems():
        wc = WordCloud(background_color="white")
        
        report_list = [report.lower() for report in cluster.values()]
        report_text = ' '.join(report_list)
        report_words = report_text.split()
        filtered_report_text = [word for word in report_words if word not in stoplist]

        # frequency hist
        fig, (a1,a2) = plt.subplots(1, 2, figsize=(15, 5))
        fdist_all = nltk.FreqDist(filtered_report_text)
        hist = [(word,freq) for word, freq in fdist_all.most_common(10)]
        words = [str(word) for word in zip(*hist)[0]]
        counts = range(len(words))
        frequency = zip(*hist)[1]
        a1.bar(counts, frequency, 1/1.5, color="blue", align='center')
        plt.sca(a1)
        plt.xticks(counts, words, rotation=45)

        # wordcloud
        wordcloud = wc.generate(' '.join(filtered_report_text))
        a2.imshow(wordcloud, interpolation='bilinear')
        a2.axis("off")
        fig.show()

        print 'Reports in ' + key, len(cluster)
        #print 'Sample report: '
        #print clusters_is["cluster{0}".format(c)][np.random.choice(len(clusters_is["cluster{0}".format(c)]))]
        print ''

def visualise_word_clusters(labels, reports, kmeans_labels, num_clusters, stoplist):
    import nltk
    import numpy as np
    import matplotlib.pyplot as plt
    from nltk.corpus import stopwords
    from wordcloud import WordCloud

    #mystoplist = ['xr', 'knee', 'both', 'x-ray', 'joint', 'left', 'right', 'space', ':']
    #stoplist = mystoplist + stopwords.words('english')

    clusters = {}
    for c in range(num_clusters):
        wc = WordCloud(background_color="white")
        #report_sample =  random.sample(list(report_list))

        #clusters_is["cluster{0}".format(k)] = [i for i,j in zip(reports, kmeans_labels_is) if j==k]
        cluster = [(i,j) for i,j,k in zip(labels, reports, kmeans_labels) if k==c]
        clusters["cluster{0}".format(c)] = dict(cluster)
        #print random.choice(list(clusters["cluster{0}".format(k)]))

        #print type(clusters["cluster{0}".format(k)])
        report_list = [report.lower() for report in clusters["cluster{0}".format(c)].values()]
        report_text = ' '.join(report_list)
        report_words = report_text.split()
        filtered_report_text = [word for word in report_words if word not in stoplist]

        # frequency hist
        fig, (a1,a2) = plt.subplots(1, 2, figsize=(15, 5))
        fdist_all = nltk.FreqDist(filtered_report_text)
        hist = [(word,freq) for word, freq in fdist_all.most_common(10)]
        words = [str(word) for word in zip(*hist)[0]]
        counts = range(len(words))
        frequency = zip(*hist)[1]
        a1.bar(counts, frequency, 1/1.5, color="blue", align='center')
        plt.sca(a1)
        plt.xticks(counts, words, rotation=45)

        # wordcloud
        wordcloud = wc.generate(' '.join(filtered_report_text))
        a2.imshow(wordcloud, interpolation='bilinear')
        a2.axis("off")
        fig.show()

        print 'Reports in cluster {0}: '.format(c), len(clusters["cluster{0}".format(c)])
        #print 'Sample report: '
        #print clusters_is["cluster{0}".format(c)][np.random.choice(len(clusters_is["cluster{0}".format(c)]))]
        print ''
        
    return clusters

def kmeans_silhouette(vectors, k_range):
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    s = []
    # b = []
    # i=0
    for n_clusters in k_range:

        print n_clusters, ' clusters'
        kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
        kmeans.fit(vectors)

        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        # bic = compute_bic(kmeans, tfidf_matrix.toarray())

        # b.append(bic)
        s.append(silhouette_score(vectors, labels, metric='euclidean'))

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
    return s