import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATHS = ["/home/jan-malte/DataLiteracyProject/union_pruned.csv", 
             "/home/jan-malte/DataLiteracyProject/union_raw.csv", 
             "/home/jan-malte/DataLiteracyProject/inter_raw.csv",
             "/home/jan-malte/DataLiteracyProject/inter_pruned.csv"]
PERPLEXITY_VALS = [10, 20, 40, 80]

def normalize_word_rows(data_array):
    word_dict = data_array
    # TODO: could probs do that with some cool double vectorized op or something, but I cant be bothered rn
    for row_idx in range(word_dict.shape[0]):
        row_sum = np.sum(word_dict[row_idx])
        normalized_row = np.vectorize(lambda x: x/row_sum)(word_dict[row_idx])
        assert len(normalized_row) == len(data_array[row_idx])
        data_array[row_idx] = normalized_row
    return data_array

def load_csv(csv_path):
    data_df = pd.read_csv(csv_path, header=0, index_col=0)
    return data_df

def get_np_dataset(data_df):
    data_handles = data_df.index.to_numpy()
    author_handles = np.vectorize(lambda x: x.split("/")[0])(data_handles)
    feature_labels = data_df.columns.to_numpy()
    data = data_df.to_numpy()
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

def plot_dim_reduced_data(reduced_data, author_handles, cat_offset=0):
    author_wise_data = {}
    for author_label in set(author_handles):
        author_mask = np.vectorize(lambda x: x == author_label)(author_handles)
        author_entries = reduced_data[author_mask]
        print(reduced_data.shape)
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

def plot_word_frequency_distribution(dataframe, word):
    author_handles, feature_labels, data = get_np_dataset(dataframe)
    data = normalize_word_rows(data)
    word_idx = np.where(feature_labels == word)
    data = np.squeeze(data[:,word_idx])
    print(np.shape(data))
    data = expand_to_2dim(data)
    plot_dim_reduced_data(data, author_handles, cat_offset=1)

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

def main():
    print("###### Top word distributions ######")
    plot_word_frequency_distributions(words=["the", "we", "of", "that"])

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
        

if __name__ == '__main__':
    main()