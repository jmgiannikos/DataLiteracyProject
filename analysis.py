import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "/home/jan-malte/DataLiteracyProject/union_pruned.csv"

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
    reduced_data = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=perplexity).fit_transform(data)
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
    data_df = load_csv(CSV_PATH)
    author_handles, feature_labels, data = get_np_dataset(data_df)
    tsne_reduced_data = tsne_dim_reduction(data)
    plot_dim_reduced_data(tsne_reduced_data, author_handles)

if __name__ == '__main__':
    main()