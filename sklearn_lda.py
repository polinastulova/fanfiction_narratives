import os
from os.path import exists, join
import numpy as np
import gzip
import requests
from pathlib import Path
from random import shuffle
from time import time
import matplotlib.pyplot as plt

from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import ParameterGrid

import joblib

#Hyperparameters LDA
n_samples = 3000
n_features = 4000
n_components = 40
batch_size = 128
max_df = 0.5
min_df = 10
init = "nndsvda"

#Hyperparameters plotting
n_top_words = 15


def download_ft_vectors():
    path = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz"
    processed_dir = join(os.getcwd(), 'data')
    Path(processed_dir).mkdir(exist_ok=True, parents=True)
    gz_file = join(processed_dir, path.split('/')[-1])
    if not exists(gz_file):
        with open(gz_file, "wb") as f:
            f.write(requests.get(path).content)
    print("Word vectors available!")
    return gz_file

def get_feature_vectors(gz_file, feature_names):
    m = []
    sorted_feature_names = []
    first_line = True
    c = 0
    with gzip.open(gz_file,'rt') as fin:
        for l in fin:
            if first_line:
                first_line = False
                continue
            fs = l.rstrip('\n').split()
            word = fs[0]
            if word in feature_names:
                vec = np.array([float(v) for v in fs[1:]])
                sorted_feature_names.append(word)
                m.append(vec)
            c+=1
            if c > 50000: #Only consider top 50k words for efficiency
                break
    return sorted_feature_names, np.array(m)


def compute_coherence(top_words, sorted_feature_list, m):
### take all the feature lists    
    feats_idx = [sorted_feature_list.index(w) for w in top_words if w in sorted_feature_list]
    truncated_m = m[feats_idx,:]
    cosines = 1-pairwise_distances(truncated_m, metric="cosine") / 2 #Dividing by 2 to avoid negative values. See https://stackoverflow.com/questions/37454785/how-to-handle-negative-values-of-cosine-similarities
    return np.mean(cosines), truncated_m, feats_idx

def return_coherence_list(model, feature_names, n_top_words, sorted_feature_list, m):
    coherences = []
    all_words_matrix = []
    topic_words = []
    for topic_idx, topic in enumerate(model.components_):
        print("---> Computing coherence for topic", topic_idx)
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        print(top_features)
        coherence, truncated_m, feats_idx  = compute_coherence(top_features, sorted_feature_list, m)
        existing_features = [sorted_feature_list[i] for i in feats_idx]
        topic_words.extend(existing_features)
        coherences.append(coherence)
        for i in range(truncated_m.shape[0]):
            all_words_matrix.append(truncated_m[i])
    all_words_matrix = np.array(all_words_matrix)
    return coherences, topic_words, all_words_matrix

def return_coherence_list_across(model, feature_names, n_top_words, sorted_feature_list, existing_topic_words, all_topic_words_m):
    cosines = 1-pairwise_distances(all_topic_words_m, metric="cosine") / 2 #Dividing by 2 to avoid negative values. See https://stackoverflow.com/questions/37454785/how-to-handle-negative-values-of-cosine-similarities
    coherences = []
    topic_words = []
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        topic_words.append(top_features)
    for i in range(len(topic_words)-1):
        for j in range(i+1,len(topic_words)):
            topic_coherences = []
            for w1 in topic_words[i]:
                for w2 in topic_words[j]:
                    if w1 in existing_topic_words and w2 in existing_topic_words:
                        idx1 = existing_topic_words.index(w1)
                        idx2 = existing_topic_words.index(w2)
                        topic_coherences.append(cosines[idx1][idx2])
            coherences.append(np.mean(topic_coherences))
    return coherences


def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(4, 5, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.3)
        ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 14})
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=8)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=14)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()

def load_data(doc_length=200):
    f = open("2000_final_no_stopwords.txt", "r", encoding = "utf-8-sig")
    docs_original = f.read()
    docs_lower = docs_original.lower().split()
    docs = [' '.join(docs_lower[start:start+doc_length]) for start in range(0, len(docs_lower),doc_length)]
    shuffle(docs)
    return docs

def get_tfs(data=None, nfeats=None, max_df=None, min_df=None):
    # Use tf (raw term count) features for LDA.
    print("Extracting tf features for LDA...")
    tf_vectorizer = CountVectorizer(
        max_df=max_df, min_df=min_df, max_features=nfeats, stop_words="english"
    )
    tfs = tf_vectorizer.fit_transform(data)
    return tf_vectorizer, tfs

def run_LDA(data=None, max_df=None, min_df=None, nfeats=None, n_components=None):
    tf_vectorizer, tfs = get_tfs(data=data, nfeats=nfeats, max_df=max_df, min_df=min_df)
    print("\nFitting LDA models with tf features, n_samples=%d and n_features=%d..." % (n_samples, nfeats))
    lda = LatentDirichletAllocation(
        n_components=n_components,
        max_iter=5,
        learning_method="online",
        learning_offset=10.0,
        random_state=0,
    )
    t0 = time()
    lda.fit(tfs)
    print("done in %0.3fs." % (time() - t0))
    perplexity = lda.perplexity(tfs)
    #print("Perplexity:",perplexity)
    #joblib.dump(lda, "lda.model")
    #joblib.dump(tf_vectorizer, "tf.model") 
    return tf_vectorizer, lda, perplexity

print("Downloading word vectors...")
gz_path = download_ft_vectors()

param_grid = {'doc_length': [100,200, 300], 'max_df': [0.5, 0.7, 0.9], 'min_df': [10, 30, 50], 'n_features': [4000], 'n_components':[40, 60, 80]}
grid = ParameterGrid(param_grid)


perplexities = []
coherences = []
coherences_across_list = []

for p in grid:
    print("\n",p)
    print("Loading dataset...")
    data = load_data(doc_length=p['doc_length'])
    data_samples = data[:n_samples]
    tf_vectorizer, lda, perplexity = run_LDA(data=data_samples, max_df=p['max_df'], min_df=p['min_df'], nfeats=p['n_features'], n_components=p['n_components'])
    tf_feature_names = tf_vectorizer.get_feature_names_out()
    sorted_feature_names, m = get_feature_vectors(gz_path, tf_feature_names)
    print("Perplexity", perplexity)
    perplexities.append(perplexity)
    cohs, existing_topic_words, all_topic_words_m = return_coherence_list(lda, tf_feature_names, n_top_words, sorted_feature_names, m)
    coherence = np.mean(cohs)
    print("COHERENCE WITHIN", coherence)
    coherences.append(coherence)
    coherences_across = np.mean(return_coherence_list_across(lda, tf_feature_names, n_top_words, sorted_feature_names, existing_topic_words, all_topic_words_m))
    coherences_across_list.append(coherences_across)
    print("COHERENCES ACROSS:",coherences_across)
    difference = [coherences[i]-coherences_across_list[i] for i in range(min(len(coherences), len(coherences_across_list)))]



best = np.argmax(difference)
p = grid[best]
print("\n\nBEST HYPERPARAMETERS:",grid[best], "COHERENCE:", coherences[best], "COHERENCE ACROSS:", coherences_across_list[best])

print(coherences)
print(coherences_across_list)
#print(difference)

#tf_vectorizer, lda, perplexity = run_LDA(data=data_samples, max_df=0.7, min_df=50, nfeats=5000, n_components=8)
#tf_vectorizer, lda, perplexity = run_LDA(data=data_samples, max_df=p['max_df'], min_df=p['min_df'], nfeats=p['n_features'], n_components=p['n_components'])
#tf_feature_names = tf_vectorizer.get_feature_names_out()
#sorted_feature_names, m = get_feature_vectors(gz_path, tf_feature_names)
#cohs = return_coherence_list(lda, tf_feature_names, n_top_words, sorted_feature_names, m)
#plot_top_words(lda, tf_feature_names, n_top_words, "Topics in LDA model"
