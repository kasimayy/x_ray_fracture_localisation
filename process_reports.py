import sys
sys.path.insert(0,'/vol/medic02/users/ag6516/miniconda/envs/PyTorch27/lib/python2.7/site-packages')
sys.path.insert(0,'/vol/medic02/users/ag6516/miniconda/envs/PyTorch27/lib/python2.7/site-packages/setuptools-27.2.0-py2.7.egg')
sys.path.insert(0,'/vol/medic02/users/ag6516/miniconda/envs/PyTorch27/lib/python2.7/site-packages/IPython/extensions')
sys.path
import csv
import json
import nltk
import numpy
from nltk.corpus import stopwords
import numpy as np
import random
random.seed(123)
np.random.seed(123)

# with open('neg_reports_tokenized.json') as data_file:
#     reports = json.load(data_file)
#
# all_findings = []
# all_impressions = []
#
# for report in reports:
#     all_findings = all_findings + report['findings']
#     all_impressions = all_impressions + report['impression']
#
# all_reports = all_impressions + all_findings

# print 'Findings: '
# print 'vocab length: ', len(sorted(set(all_findings)))
# print 'avg no. of appearances: ', len(all_findings)/len(sorted(set(all_findings)))
# fdist_findings = nltk.FreqDist(all_findings)
# freqs = [freq for _, freq in fdist_findings.most_common()]
# print 'STD:', numpy.std(freqs)
# print fdist_findings.hapaxes()
# for word, frequency in fdist_findings.most_common(50):
#     print(u'{};{}'.format(word, frequency))

# print '\n Impressions: '
# print 'vocab length: ', len(sorted(set(all_impressions)))
# print 'avg no. of appearances: ', len(all_impressions)/len(sorted(set(all_impressions)))
# fdist_impressions = nltk.FreqDist(all_impressions)
# freqs = [freq for _, freq in fdist_impressions.most_common()]
# print 'STD:', numpy.std(freqs)
# print fdist_impressions.hapaxes()
# for word, frequency in fdist_impressions.most_common(50):
#     print(u'{};{}'.format(word, frequency))

# print '\n All: '
# print 'vocab length: ', len(sorted(set(all_reports)))
# print 'avg no. of appearances: ', len(all_reports)/len(sorted(set(all_reports)))
# fdist_all = nltk.FreqDist(all_reports)
# freqs = [freq for _, freq in fdist_all.most_common()]
# print 'STD:', numpy.std(freqs)
# print fdist_all.hapaxes()
# for word, frequency in fdist_all.most_common(50):
#     print(u'{};{}'.format(word, frequency))

d_labeld_data = []
with open('disease_labeled_reports.txt', 'r') as df:
    reader = csv.reader(df, delimiter='|')
    for row in reader:
        #print row
        d_labeld_data.append(row)

print len(d_labeld_data)
all_terms = []
for report in d_labeld_data:
    for i in range(len(report)):
        if i != 0:
            all_terms.append(report[i])

#print len(all_terms)
#print 'vocab length: ', len(sorted(set(all_terms)))
#print 'avg no. of appearances: ', len(all_terms)/len(sorted(set(all_terms)))
fdist_all = nltk.FreqDist(all_terms)
freqs = [freq for _, freq in fdist_all.most_common()]
#print 'STD:', numpy.std(freqs)
#print fdist_all.hapaxes()
labels = []
for word, frequency in fdist_all.most_common(100):
    count = 0
    for report in d_labeld_data:
        if word in ' '.join(report):
            if len(report) > 2:
                count+=1
    #print(u'{};{};{}'.format(word, frequency,count))
    if frequency >= 30 and float(count)/float(frequency)*100 < 100:
        if word != 'disease' and word != 'mas' and word != '':
            labels.append(word)

print len(labels), labels

# assign labels from list to images
d_labeld_images = []
all_labels = []
for report in d_labeld_data:
    terms = [e for e in labels if e in '\n'.join(report)]
    if len(terms) == 1:
        d_labeld_images.append([report[0], terms[0]])
        all_labels.append(terms[0])
        #print terms

print len(d_labeld_images)

fdist_all = nltk.FreqDist(all_labels)
freqs = [freq for _, freq in fdist_all.most_common()]

labels_f_list = []
onehot_indexes = []
i=0
for word, frequency in fdist_all.most_common(len(labels)):
    onehot_indexes.append([word, i])
    i+=1
    labels_f_list.append(frequency)
    # print(u'{};{}'.format(word, frequency))

# Split into training and testing, save as csv
tr_portion = 0.8
val_portion = 0.1
te_portion = 0.1
tr_data = []
te_data = []
val_data = []

for label, index in onehot_indexes:
    dfi = []
    for report in d_labeld_images:
        if label in report[1]:
            dfi.append([report[0], index])

    if len(dfi)<10:
        continue
    dfi_len_val = int(np.round(len(dfi)*0.1))
    dfi_len_te = int(np.round(len(dfi)*0.1))
    dfi_len_tr = len(dfi) - int(np.round(len(dfi)*0.2))

    tr_data += [dfi[i] for i in range(0, dfi_len_tr)]
    te_data += [dfi[i] for i in range(dfi_len_tr, dfi_len_tr+dfi_len_te)]
    val_data += [dfi[i] for i in range(dfi_len_tr+dfi_len_te, len(dfi))]


print(len(tr_data), len(val_data), len(te_data))
print len(tr_data) + len(val_data) + len(te_data)
# calculate ratios of 'normal' cases to max and min number of other unique mesh term combinations
counts_normal = float(labels_f_list[0])
counts_max = float(labels_f_list[1])
counts_min = float(labels_f_list[-1])

default_aug_ratio = 4.0
normal_max_ratio = default_aug_ratio*counts_max/counts_normal
max_min_ratio = 1.0*counts_max/counts_min
normal_min_ratio = 1.0*counts_normal/counts_min
print 'normal_max_ratio:', normal_max_ratio
print 'max_min_ratio:', max_min_ratio
print 'normal_min_ratio:', normal_min_ratio

# Augment training data
tr_data_aug = []
te_data_aug = []
val_data_aug = []
do_aug = False
for _, i in onehot_indexes:
    dfi = []
    for report, label in tr_data:
        if label == i:
            dfi.append(report)

    for report in dfi:
        if i == 0:
            do_aug = False
            tr_data_aug.append([report, str(i), int(do_aug)])
        else:
            for j in range(int(np.round(1.0*default_aug_ratio*counts_max/labels_f_list[i]))):
                if j == 0:
                    do_aug = False
                else:
                    do_aug = True
                tr_data_aug.append([report, str(i), int(do_aug)])
    do_aug = False
    dfi = []
    for report, label in te_data:
        if label == i:
            dfi.append(report)
    for report in dfi:
        te_data_aug.append([report, str(i), int(do_aug)])

    dfi = []
    for report, label in val_data:
        if label == i:
            dfi.append(report)
    for report in dfi:
        val_data_aug.append([report, str(i), int(do_aug)])

print "Augmented data count:"
print len(tr_data_aug), len(te_data_aug), len(val_data_aug)

# Calculate weights for classes
weights = [round(sum(labels_f_list)/float(f),2) for f in labels_f_list]
print "Class weights:"
print weights

import glob
# shuffle train/test/split and output into csv
random.shuffle(tr_data_aug)
random.shuffle(te_data_aug)
random.shuffle(val_data_aug)
with open('processed/tr_data_aug.csv', 'wb') as csvf:
    csvw = csv.writer(csvf, delimiter=' ')
    for row in tr_data_aug:
        csvw.writerow(row)
with open('processed/te_data_aug.csv', 'wb') as csvf:
    csvw = csv.writer(csvf, delimiter=' ')
    for row in te_data_aug:
        csvw.writerow(row)
with open('processed/val_data_aug.csv', 'wb') as csvf:
    csvw = csv.writer(csvf, delimiter=' ')
    for row in val_data_aug:
        csvw.writerow(row)
with open('processed/loss_weights.csv', 'wb') as csvf:
    csvw = csv.writer(csvf, delimiter=' ')
    csvw.writerow(weights)
