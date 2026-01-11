import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from PyDictionary import PyDictionary
from itertools import chain
from nltk.corpus import wordnet
import nltk

CSV_PATHS = ["/home/jan-malte/DataLiteracyProject/union_pruned.csv",
             "/home/jan-malte/DataLiteracyProject/union_raw.csv", 
             "/home/jan-malte/DataLiteracyProject/inter_raw.csv",
             "/home/jan-malte/DataLiteracyProject/inter_pruned.csv"]
PERPLEXITY_VALS = [10, 20, 40, 80]

# NOTE: somethings fucked here. produces very weird results
def normalize_word_rows(data_array):
    word_dict = data_array
    # TODO: could probs do that with some cool double vectorized op or something, but I cant be bothered rn
    normalized_rows = []
    for row_idx in range(word_dict.shape[0]):
        row_sum = np.sum(word_dict[row_idx])
        normalized_row = np.vectorize(lambda x: x/row_sum)(word_dict[row_idx])
        assert len(normalized_row) == len(data_array[row_idx])
        normalized_rows.append(normalized_row)
    normalized_data_array = np.vstack(normalized_rows)
    return normalized_data_array

# TODO: Weird duplication glitch happens here. I dont know why. load_csv now filters for it, but we should find the source of the error
# TODO: same situation with the zero word documents that show up
# TODO: also there seem to be word columns with zero occurences? Something in data collection is very wrong. Does not come from dropping duplicates or pruning zero count rows. Happens in dataset generation? Current guess is a bug during csv writing, where rows are overridden somehow
def load_csv(csv_path):
    data_df = pd.read_csv(csv_path, header=0, index_col=0)
    data_df.drop_duplicates(inplace=True)
    data_array = data_df.to_numpy()
    to_prune_indices = []
    for i, index in enumerate(data_df.index):
        row = data_array[i,:]
        row_sum = np.sum(row)
        if row_sum == 0:
            to_prune_indices.append(index)
    for index in to_prune_indices:
        data_df.drop(index, axis=0, inplace=True)
    return data_df

def get_np_dataset(data_df):
    data_handles = data_df.index.to_numpy()
    author_handles = np.vectorize(lambda x: x.split("/")[0])(data_handles)
    feature_labels = data_df.columns.to_numpy()
    data = data_df.to_numpy()

    # TODO: This shouldnt be needed. Dataset should contain no words with zero occurences
    zero_occurence_words_mask = np.sum(data, axis=0) > 0
    data = data[:,zero_occurence_words_mask]
    feature_labels = feature_labels[zero_occurence_words_mask]

    # check that data has expected shape of (samples, features)
    assert data.shape[0] == len(author_handles) and data.shape[1] == len(feature_labels)
    assert len(feature_labels.shape) == 1 and len(author_handles.shape)
    return author_handles, feature_labels, data

def tsne_dim_reduction(data, perplexity=30):
    if isinstance(perplexity, list):
        reduced_data = []
        for perplexity_value in perplexity:
            reduced_data.append(tsne_dim_reduction(data, perplexity=perplexity_value))
    else: 
        reduced_data = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=perplexity).fit_transform(data)
    return reduced_data

def pca_dim_reduction(data, axes=2):
    pca = PCA(n_components=axes)
    reduced_data = pca.fit_transform(data)
    loadings = pca.components_
    explained_variance = pca.explained_variance_

    return reduced_data, loadings, explained_variance

def get_author_entries(author_handles, author_label, data):
    author_mask = np.vectorize(lambda x: x == author_label)(author_handles)
    author_entries = data[author_mask]
    return author_entries, author_mask

def plot_dim_reduced_data(reduced_data, author_handles, cat_offset=0):
    author_wise_data = {}
    for author_label in set(author_handles):
        author_entries, author_mask = get_author_entries(author_handles, author_label, reduced_data)
        assert author_entries.shape == (np.sum(author_mask), reduced_data.shape[1])
        author_wise_data[author_label] = author_entries
    
    for i, author_label in enumerate(author_wise_data.keys()):
        y_offset = i*cat_offset
        x = author_wise_data[author_label][:,0].T
        y = author_wise_data[author_label][:,1].T + y_offset
        plt.scatter(x, y)
    plt.show()

def words_only(data_array):
    return data_array[:, :-2]

def get_topn_words(values, words):
    sorted_zip = list(zip(values, words))
    sorted_zip.sort(key=lambda x: abs(x[0]), reverse=True)
    values = [tup[0] for tup in sorted_zip]
    words = [tup[1] for tup in sorted_zip]
    return values, words

def plot_loadings(loadings, word_list, max_show=20):
    for loading in loadings:
        loading_1, word_list_1 = get_topn_words(loading, word_list)
        plt.bar(word_list_1[:max_show], loading_1[:max_show])
        plt.show()

# adds randomized second dimension for plotting purposes
def expand_to_2dim(data):
    if len(np.shape(data)) == 1:
        data = np.expand_dims(data,1)
    random_ys = np.random.permutation(len(data))/len(data)
    data = np.append(data, np.expand_dims(random_ys,1), axis=1)
    return data

def process_PCA(csv_paths=CSV_PATHS, dims=2):
    for csv_path in csv_paths:
        data_df = load_csv(csv_path)
        author_handles, feature_labels, data = get_np_dataset(data_df)
        data = normalize_word_rows(data)
        data = words_only(data) # fetch only words, so that all entries have the same unit (word frequency)
        pca_reduced_data, loadings, explained_variance = pca_dim_reduction(data, axes=dims)
        if dims == 1:
            pca_reduced_data = expand_to_2dim(pca_reduced_data)
            cat_offset = 1
        else:
            cat_offset = 0
        print(explained_variance)
        print(f"CSV: {csv_path.split("/")[-1]}")
        plot_dim_reduced_data(pca_reduced_data, author_handles, cat_offset=cat_offset)
        plot_loadings(loadings, feature_labels)

def plot_binned_barchart(data, n_bins, author_handles, ylabel = "frequency"):
    min_val = np.min(data, axis=None)
    max_val = np.max(data, axis=None)

    value_range = (max_val-min_val)
    bin_size = value_range/n_bins

    bins = np.arange(start=min_val, stop=max_val, step=bin_size)[:-1]
    data_series_list = []
    for author_label in set(author_handles):
        masked_data, mask = get_author_entries(author_handles, author_label, data)
        local_series = [0]*len(bins)
        for masked_data_value in masked_data.flatten():
            for bin_idx, bin_val in enumerate(bins):
                if masked_data_value > bin_val and masked_data_value <= bin_val+bin_size:
                    local_series[bin_idx] += 1
        local_series = [local_series_val/np.sum(mask) for local_series_val in local_series]
        data_series_list.append(local_series)
    data_series_array = np.array(data_series_list)

    x = np.arange(data_series_array.shape[1])
    width = 1/(data_series_array.shape[0]+1)
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for i, frequency in enumerate(data_series_array):
        offset = width * multiplier
        rects = ax.bar(x+offset, frequency, width, label=list(set(author_handles))[i])
        ax.bar_label(rects, padding=3)
        multiplier += 1
    
    ax.set_ylabel(ylabel)
    ax.set_xticks(ticks = x+width, labels = ["{:e}".format(entry)[:3] + "e" +"{:e}".format(entry).split("e")[-1] for entry in bins])
    ax.legend()
    plt.show()

def plot_word_frequency_distribution(dataframe, word, n_bins=10):
    author_handles, feature_labels, data = get_np_dataset(dataframe)
    data = normalize_word_rows(data)
    print(np.max(data))
    word_idx = np.where(feature_labels == word)
    data = np.squeeze(data[:,word_idx])
    print(data)
    print(data.shape)
    print(np.sum(data))
    plot_binned_barchart(data, 10, author_handles)

def plot_word_frequency_distributions(words, csv_paths=CSV_PATHS):
    for csv_path in csv_paths:
        print(f"CSV: {csv_path.split("/")[-1]}")
        data_df = load_csv(csv_path)
        for word in words:
            print(f"#### Distribution over '{word}' frequency ####")
            plot_word_frequency_distribution(data_df, word)

def plot_sentence_length_distribution(sentence_lengths):
    length_dict = {}
    # init lenght dict
    max_sentence_length = max(sentence_lengths)
    for i in range(max_sentence_length+1):
        length_dict[i] = 0
    for sentence_len in sentence_lengths:
        length_dict[sentence_len] += 1
    plt.bar(length_dict.keys(), length_dict.values()) # for now im just gonna hope this works
    plt.show()

def plot_common_words_frequencies(csv_paths=CSV_PATHS, commonality_threshhold=0.5):
    relevant_csv_paths = csv_paths
    for csv_path in relevant_csv_paths:
        print(f"CSV: {csv_path.split("/")[-1]}")
        data_df = load_csv(csv_path)
        common_words = get_common_words(data_df, commonality_thershhold=commonality_threshhold)
        for word in common_words:
            print(f"#### Distribution over '{word}' frequency ####")
            plot_word_frequency_distribution(data_df, word)

def get_common_words(union_word_hist_dataframe, commonality_thershhold=0.5):
    common_words = []
    recorded_documents, _ = union_word_hist_dataframe.shape
    for word in union_word_hist_dataframe.columns:
        counts = union_word_hist_dataframe[word]
        if sum([1 if count>0 else 0 for count in counts])/recorded_documents > commonality_thershhold:
            common_words.append(word)
    return common_words

def pca_and_tsne():
    print("###### PCA onto one Dim ######")
    process_PCA(dims=1)
    
    print("###### PCA ######")
    process_PCA()

    print("###### t-SNE ######")
    for csv_path in CSV_PATHS:
        data_df = load_csv(csv_path)
        author_handles, feature_labels, data = get_np_dataset(data_df)
        data = normalize_word_rows(data)
        tsne_reduced_data_list = tsne_dim_reduction(data, PERPLEXITY_VALS)
        for i, tsne_reduced_data in enumerate(tsne_reduced_data_list):
            print(f"CSV: {csv_path.split("/")[-1]}, PERPLEXITY: {PERPLEXITY_VALS[i]}")
            plot_dim_reduced_data(tsne_reduced_data, author_handles)

def prune_synonyms(synonyms, dataframe):
    print(synonyms)
    retained_synonyms = []
    recorded_words = dataframe.columns
    for synonym in synonyms:
        if synonym in recorded_words:
            retained_synonyms.append(synonym)
    return retained_synonyms

def plot_author_wise_synonym_prefernece(synonyms, dataframe):
    synonyms = prune_synonyms(synonyms, dataframe)
    if len(synonyms) > 1:
        author_handles, feature_labels, data = get_np_dataset(dataframe)
        authors = list(set(author_handles))
        author_preference_distribution = []
        for author in authors:
            relevant_documents, _ = get_author_entries(author_handles, author, data)
            synonym_distribution = []
            for synonym in synonyms:
                synonym_idx = np.where(feature_labels == synonym)
                synonym_count = np.sum(np.squeeze(relevant_documents[:,synonym_idx]))
                synonym_distribution.append(synonym_count)
            total_synonyms_count = sum(synonym_distribution)
            synonym_distribution = [synonym_count/total_synonyms_count for synonym_count in synonym_distribution]
            author_preference_distribution.append(synonym_distribution)
        author_preference_distribution = np.array(author_preference_distribution).T

        width = 0.5
        fig, ax = plt.subplots()
        bottom = np.zeros(3)

        for i, word_prevalence in enumerate(author_preference_distribution):
            p = ax.bar(authors, word_prevalence, width, label=synonyms[i], bottom=bottom)
            bottom += word_prevalence
        ax.legend(loc="upper right")
        plt.show()

def synonym_analysis(dataframe, words):
    processed_words = []
    for word in words:
        if word not in processed_words:
            synonyms = wordnet.synsets(word)
            print(synonyms)
            synonyms = list(set(chain.from_iterable([word.lemma_names() for word in synonyms])))
            processed_words = processed_words + synonyms
            #synonyms.append(word)
            if len(synonyms) > 0:
                plot_author_wise_synonym_prefernece(synonyms, dataframe)

def main():
    #print("###### Top word distributions ######")
    #plot_word_frequency_distributions(words=["the", "we", "of", "that"])
    print("###### Plot synonym preferences ######")
    histogram = load_csv("/home/jan-malte/DataLiteracyProject/union_pruned.csv")
    checked_words = get_common_words(histogram, commonality_thershhold=0.9)
    synonym_analysis(histogram, checked_words)
    print("###### Plot common word frequencies ######")
    plot_common_words_frequencies(commonality_threshhold=0.8)
    

if __name__ == '__main__':
    nltk.download('wordnet')

    main()