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
    word_dict = data_array[:,:-2]
    # TODO: could probs do that with some cool double vectorized op or something, but I cant be bothered rn
    for row_idx in range(word_dict.shape[0]):
        row_sum = np.sum(word_dict[row_idx])
        normalized_row = np.vectorize(lambda x: x/row_sum)(word_dict[row_idx])
        assert len(normalized_row) == len(data_array[row_idx]) - 2
        data_array[row_idx, :-2] = normalized_row
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

def pca_dim_reduction(data):
    reduced_data = PCA(n_components=2).fit_transform(data)
    return reduced_data

def plot_dim_reduced_data(reduced_data, author_handles):
    author_wise_data = {}
    for author_label in set(author_handles):
        author_mask = np.vectorize(lambda x: x == author_label)(author_handles)
        author_entries = reduced_data[author_mask]
        print(reduced_data.shape)
        assert author_entries.shape == (np.sum(author_mask), reduced_data.shape[1])
        author_wise_data[author_label] = author_entries
    
    for author_label in author_wise_data.keys():
        x = author_wise_data[author_label][:,0].T
        y = author_wise_data[author_label][:,1].T
        plt.scatter(x, y)
    plt.show()

def main():
    print("###### PCA ######")
    for csv_path in CSV_PATHS:
        data_df = load_csv(csv_path)
        author_handles, feature_labels, data = get_np_dataset(data_df)
        data = normalize_word_rows(data)
        pca_reduced_data = pca_dim_reduction(data)
        print(f"CSV: {csv_path.split("/")[-1]}")
        plot_dim_reduced_data(pca_reduced_data, author_handles)

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