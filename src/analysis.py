"""
Analysis and visualization module for extracted features.
Provides t-SNE, PCA, and plotting functions for word frequency analysis.

Source: jan-analysis/analysis.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from typing import List, Tuple, Optional, Dict, Set
import logging
import textstat
from scipy.spatial import distance
import json
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================
def load_metadata(csv_path: str) -> pd.DataFrame:
    data_df = pd.read_csv(csv_path, header=0, index_col=0)
    numpy_data = data_df.to_numpy()
    col_names = data_df.columns
    index = data_df.index
    new_index = []
    # reindex to be in line with word_hist csvs and sentence lenght jsons
    for file_id in index:
        new_index.append(file_id.replace("/", "_"))
    data_df = pd.DataFrame(data=numpy_data, index=new_index, columns=col_names)
    return data_df

def load_sentence_json(json_path: str, max_len=-1) -> Tuple[pd.DataFrame, dict]:
    """
    Load sentence length json.

    Args:
        json_path: Path to json file with sentence lengths
        max_sent_len: ignores all sentence lenghts that are longer than max_sent_len. -1 means no filtering

    Returns:
        Cleaned DataFrame with documents as rows and words as columns

    Source: jan-analysis/analysis.py
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    index = []
    max_val = 0
    for file_id in data.keys():
        index.append(file_id)
        if max(data[file_id]) > max_val:
            max_val = max(data[file_id])
    
    if max_len > 0:
        max_val = max_len

    rows = []
    for file_id in data.keys():
        row = np.zeros((max_val,))
        for value in data[file_id]:
            # assume min value is 1 (no empty sentences). shift index down appropriately.
            if (max_len > 0 and value <= max_len) or max_len <= 0:
                row[value-1] += 1

        rows.append(row)
    
    np_data = np.vstack(rows)
    sentence_df = pd.DataFrame(data=np_data, index=index, columns=range(1, max_val+1))
    return sentence_df, data

def load_csv(csv_path: str) -> pd.DataFrame:
    """
    Load word histogram CSV with deduplication and zero-row removal.

    Args:
        csv_path: Path to CSV file with word histograms

    Returns:
        Cleaned DataFrame with documents as rows and words as columns

    Source: jan-analysis/analysis.py
    """
    data_df = pd.read_csv(csv_path, header=0, index_col=0)
    data_df.drop_duplicates(inplace=True)

    # Remove rows with zero total word count
    # TODO: this could cause issue when combining with sentence len stats. Expect the same set of documents
    data_array = data_df.to_numpy()
    to_prune = []
    for i, index in enumerate(data_df.index):
        if np.sum(data_array[i, :]) == 0:
            to_prune.append(index)

    for index in to_prune:
        data_df.drop(index, axis=0, inplace=True)
        logger.warning(f"Removed zero-count row: {index}")

    return data_df


def get_np_dataset(data_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract numpy arrays from dataframe.

    Args:
        data_df: DataFrame with documents as rows and words as columns

    Returns:
        Tuple of (author_handles, feature_labels, data_matrix)

    Source: jan-analysis/analysis.py
    """
    data_handles = data_df.index.to_numpy()
    # Extract author from handle (assumes format "author/paper_id")
    author_handles = np.vectorize(lambda x: x.split("/")[0] if "/" in x else x)(data_handles)
    feature_labels = data_df.columns.to_numpy()
    data = data_df.to_numpy()

    # Remove zero-occurrence words
    mask = np.sum(data, axis=0) > 0
    if not np.all(mask):
        removed_count = np.sum(~mask)
        logger.debug(f"Removed {removed_count} zero-occurrence word columns")
        data = data[:, mask]
        feature_labels = feature_labels[mask]

    return author_handles, feature_labels, data


def normalize_word_rows(data_array: np.ndarray) -> np.ndarray:
    """
    Normalize each row to sum to 1 (relative frequencies).

    Args:
        data_array: Word frequency matrix (documents x words)

    Returns:
        Normalized matrix where each row sums to 1

    Source: jan-analysis/analysis.py
    """
    row_sums = np.sum(data_array, axis=1, keepdims=True)
    # Avoid division by zero
    row_sums = np.where(row_sums == 0, 1, row_sums)
    return data_array / row_sums

# =============================================================================
# SUPPLEMENTARY FEATURE EXTRACTION
# =============================================================================

def get_common_words(word_hist_dataframe, commonality_thershhold=0.5):
    common_words = []
    recorded_documents, _ = word_hist_dataframe.shape
    for word in word_hist_dataframe.columns:
        counts = word_hist_dataframe[word]
        if sum([1 if count>0 else 0 for count in counts])/recorded_documents > commonality_thershhold:
            common_words.append(word)
    return common_words

def get_common_word_df(words_df, commonality_thershhold=0.5):
    common_words = get_common_words(words_df, commonality_thershhold)
    return words_df[common_words]

def get_mean_and_stdev_sent(sentence_df, max_len=-1):
    """
    Takes a non-normalized sentence length dataframe and returns a dataframe containing mean and
    stdev sentence length

    Args:
        words_df: a dataframe containing sentence lenght counts. Row index must be document identifiers, col
                  must be sentence lengths.
        max_len:  set to affect the largest considered sentence length. default to -1 which means no restriction.

    Returns:
        a dataframe two columns: mean sentence lenght and stdev sentence length
    """
    if max_len > 0:
        data = sentence_df.to_numpy()[:max_len]
    else:
        data = sentence_df.to_numpy()
    
    stdevs = np.std(data, axis=1, keepdims=True)
    means = np.mean(data, axis=1, keepdims=True)

    np_data = np.hstack((means, stdevs))
    result_df = pd.DataFrame(data=np_data, index=sentence_df.index, columns=["mean", "stdev"])
    return result_df


# NOTE: THE DF GIVEN TO THIS SHOULD BE NON-NORMALIZED!
# TODO: implement first double loop more efficiently. Slow af
def get_syllable_counts(words_df):
    """
    Takes a non-normalized word histogram dataframe and computes the syllable count distribution

    Args:
        words_df: a dataframe containing word counts. Row index must be document identifiers, col
                  must be words.

    Returns:
        a dataframe with syllable counts as col index and document identifiers as row idndex
    """
    row_index_list = []
    words = words_df.columns
    row_dicts = []
    max_syllable_count = 0
    for identifier, row in words_df.iterrows():
        row_index_list.append(identifier)
        row_dict = {}
        for word_idx, word_count in enumerate(row):
            word = words[word_idx]
            syllable_count = textstat.syllable_count(word)
            if syllable_count > max_syllable_count:
                max_syllable_count = syllable_count
            if syllable_count in row_dict.keys():
                row_dict[syllable_count] += word_count
            else:
                row_dict[syllable_count] = word_count
        row_dicts.append(row_dict)

    for row_idx, _ in enumerate(row_dicts):
        row = np.zeros((1,max_syllable_count))
        for syllable_count in row_dicts[row_idx].keys():
            # reduce synonym_count by one so it can serve as index for the row
            row[0, syllable_count-1] = row_dicts[row_idx][syllable_count]
        if row_idx == 0:
            data_array = row
        else:
            data_array = np.vstack((data_array, row))
    
    syllable_count_df = pd.DataFrame(data=data_array, index=row_index_list, columns=range(1, max_syllable_count+1))
    return syllable_count_df

# NOTE: the textstats function is_difficult_word checks if the word is in the Dale-Chall list of easy words or not. However,
#       the authors of said library note that the function does NOT check for regular inflections of easy words. We could 
#       potentially improve the accuracy of this metric by stemming the words in the histogram first. (Stemming may be a good idea
#       in general)           
def get_easy_words_count(words_df):
    """
    Takes a non-normalized word histogram dataframe and counts the number of words on the Dale-Chall list of easy words

    Args:
        words_df: a dataframe containing word counts. Row index must be document identifiers, col
                  must be words.

    Returns:
        a dataframe containing easy word count and ratio as two columns
    """
    is_easy_mask = [0]*len(words_df.columns)
    for col_idx, word in enumerate(words_df.columns): 
        if textstat.is_easy_word(word):
            is_easy_mask[col_idx] = 1
    is_easy_mask = np.array(is_easy_mask)
    
    rows = []
    row_index = []
    for row_id, row in words_df.iterrows():
        easy_word_count = np.inner(row.to_numpy(), is_easy_mask)
        easy_word_ratio = easy_word_count/np.sum(row.to_numpy())
        rows.append([easy_word_count, easy_word_ratio])
        row_index.append(row_id)
    rows_np = np.array(rows)
    
    easy_word_count_df = pd.DataFrame(data=rows_np, index=row_index, columns=["easy_word_count", "easy_word_ratio"])
    return easy_word_count_df

def get_monosyllabic_words(word_df, is_syllabic=False):
    """
    Takes a non-normalized word histogram dataframe or a syllable count df and computes total and relative monosyllabic
    word count.

    Args:
        words_df: a dataframe containing word counts. Row index must be document identifiers, col
                  must be words. alternatively the result df of calling get_syllable_counts
        is_syllabic: set to True if passing a syllable count dataframe, otherwise set to false

    Returns:
        a dataframe containing monosyllabic count and ratio
    """
    if not is_syllabic:
        word_df = get_syllable_counts(word_df)
    monosyllabic_count_array = word_df[1].to_numpy()
    monosyllabic_ratio = monosyllabic_count_array / np.sum(word_df.to_numpy(), axis=1, keepdims=False)
    cols_np = np.hstack((np.expand_dims(monosyllabic_count_array,1), np.expand_dims(monosyllabic_ratio,1)))
    monosyllabic_count_df = pd.DataFrame(data=cols_np, index=word_df.index, columns=["monosyl_count", "monosyl_ratio"])
    return monosyllabic_count_df
   

# =============================================================================
# DIMENSIONALITY REDUCTION FUNCTIONS
# =============================================================================

def tsne_dim_reduction(
    data: np.ndarray,
    perplexity: int = 30,
    n_components: int = 2,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Apply t-SNE dimensionality reduction.

    Args:
        data: Input data matrix (samples x features)
        perplexity: t-SNE perplexity parameter
        n_components: Number of output dimensions
        random_state: Random seed for reproducibility

    Returns:
        Reduced data matrix (samples x n_components)

    Source: jan-analysis/analysis.py
    """
    tsne = TSNE(
        n_components=n_components,
        learning_rate='auto',
        init='pca',
        perplexity=perplexity,
        random_state=random_state
    )
    return tsne.fit_transform(data)


def pca_dim_reduction(
    data: np.ndarray,
    n_components: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply PCA dimensionality reduction.

    Args:
        data: Input data matrix (samples x features)
        n_components: Number of principal components

    Returns:
        Tuple of (reduced_data, loadings, explained_variance)

    Source: jan-analysis/analysis.py
    """
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    return reduced_data, pca.components_, pca.explained_variance_


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def get_author_entries(
    author_handles: np.ndarray,
    author_label: str,
    data: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract data entries for a specific author.

    Args:
        author_handles: Array of author labels
        author_label: Author to filter for
        data: Data matrix

    Returns:
        Tuple of (author_data, boolean_mask)
    """
    author_mask = author_handles == author_label
    author_entries = data[author_mask]
    return author_entries, author_mask


def plot_dim_reduced_data(
    reduced_data: np.ndarray,
    author_handles: np.ndarray,
    title: str = "Dimensionality Reduced Data",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Plot 2D dimensionality-reduced data colored by author.

    Args:
        reduced_data: 2D data matrix (samples x 2)
        author_handles: Array of author labels for each sample
        title: Plot title
        save_path: If provided, save figure to this path
        figsize: Figure size tuple

    Source: jan-analysis/analysis.py
    """
    plt.figure(figsize=figsize)

    unique_authors = list(set(author_handles))
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(unique_authors), 10)))

    for i, author in enumerate(unique_authors):
        author_data, _ = get_author_entries(author_handles, author, reduced_data)
        plt.scatter(
            author_data[:, 0],
            author_data[:, 1],
            c=[colors[i % 10]],
            label=author,
            alpha=0.7
        )

    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    else:
        plt.show()

    plt.close()


def get_topn_words(
    values: np.ndarray,
    words: np.ndarray,
    n: int = 20
) -> Tuple[List[float], List[str]]:
    """
    Get top N words by absolute value.

    Args:
        values: Array of values (e.g., PCA loadings)
        words: Array of word labels
        n: Number of top words to return

    Returns:
        Tuple of (sorted_values, sorted_words)
    """
    sorted_pairs = sorted(zip(values, words), key=lambda x: abs(x[0]), reverse=True)
    values_sorted = [p[0] for p in sorted_pairs[:n]]
    words_sorted = [p[1] for p in sorted_pairs[:n]]
    return values_sorted, words_sorted


def plot_loadings(
    loadings: np.ndarray,
    word_list: np.ndarray,
    max_show: int = 20,
    save_path: Optional[str] = None
) -> None:
    """
    Plot PCA loadings (most important words per component).

    Args:
        loadings: PCA components matrix (n_components x n_features)
        word_list: Array of word labels
        max_show: Maximum number of words to show per component
        save_path: Base path for saving figures (component number appended)

    Source: jan-analysis/analysis.py
    """
    for comp_idx, loading in enumerate(loadings):
        loading_sorted, words_sorted = get_topn_words(loading, word_list, max_show)

        plt.figure(figsize=(12, 6))
        colors = ['green' if v > 0 else 'red' for v in loading_sorted]
        plt.bar(range(len(words_sorted)), loading_sorted, color=colors)
        plt.xticks(range(len(words_sorted)), words_sorted, rotation=45, ha='right')
        plt.title(f"PCA Component {comp_idx + 1} Loadings")
        plt.xlabel("Word")
        plt.ylabel("Loading")
        plt.tight_layout()

        if save_path:
            plt.savefig(f"{save_path}_comp{comp_idx + 1}.png", dpi=150, bbox_inches='tight')
            logger.info(f"Saved loading plot to {save_path}_comp{comp_idx + 1}.png")
        else:
            plt.show()

        plt.close()


def plot_binned_barchart(
    data: np.ndarray,
    n_bins: int,
    author_handles: np.ndarray,
    ylabel: str = "Frequency",
    title: str = "Distribution by Author",
    save_path: Optional[str] = None
) -> None:
    """
    Plot binned bar chart showing distribution across authors.

    Args:
        data: 1D data array
        n_bins: Number of histogram bins
        author_handles: Author labels corresponding to data
        ylabel: Y-axis label
        title: Plot title
        save_path: If provided, save figure to this path

    Source: jan-analysis/analysis.py
    """
    min_val = np.min(data)
    max_val = np.max(data)
    value_range = max_val - min_val

    if value_range == 0:
        logger.warning("Data has zero range, cannot create binned chart")
        return

    bin_size = value_range / n_bins
    bins = np.arange(start=min_val, stop=max_val, step=bin_size)

    unique_authors = list(set(author_handles))
    data_series_list = []

    for author_label in unique_authors:
        masked_data, mask = get_author_entries(author_handles, author_label, data.reshape(-1, 1))
        local_series = [0] * len(bins)

        for value in masked_data.flatten():
            for bin_idx, bin_val in enumerate(bins):
                if bin_val <= value < bin_val + bin_size:
                    local_series[bin_idx] += 1
                    break

        # Normalize by number of samples for this author
        n_samples = np.sum(mask)
        if n_samples > 0:
            local_series = [v / n_samples for v in local_series]

        data_series_list.append(local_series)

    data_series_array = np.array(data_series_list)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(bins))
    width = 0.8 / len(unique_authors)

    for i, (author, series) in enumerate(zip(unique_authors, data_series_array)):
        offset = width * i
        ax.bar(x + offset, series, width, label=author, alpha=0.8)

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Value bins")
    ax.set_title(title)
    ax.set_xticks(x + width * len(unique_authors) / 2)
    ax.set_xticklabels([f"{b:.2e}" for b in bins], rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved bar chart to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_sentence_length_distribution(
    sentence_lengths: Dict[str, List[int]],
    title: str = "Sentence Length Distribution",
    save_path: Optional[str] = None
) -> None:
    """
    Plot sentence length distribution for multiple documents/authors.

    Args:
        sentence_lengths: Dictionary mapping document/author ID to list of sentence lengths
        title: Plot title
        save_path: If provided, save figure to this path
    """
    plt.figure(figsize=(12, 6))

    for doc_id, lengths in sentence_lengths.items():
        if lengths:
            plt.hist(lengths, bins=30, alpha=0.5, label=doc_id)

    plt.xlabel("Sentence Length (words)")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved distribution plot to {save_path}")
    else:
        plt.show()

    plt.close()

def visualize_feature_analysis(data_df, title, save_path):
    ax = sns.heatmap(data_df)
    ax.set_title(title)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved distribution plot to {save_path}")
    else:
        plt.show()


# =============================================================================
# UTIL FUNCTIONS
# =============================================================================

def get_groupings(metadata_df, group_cols):
    groupings_dict = {}
    for group_col in group_cols:
        # check if group col is a valid column
        if group_col in metadata_df.columns:
            cleaned_metadata_df = metadata_df.dropna(subset=[group_col]) # ignore all rows that contain nan in the relevant column
            group_dict = {}
            for file_id, _ in cleaned_metadata_df.iterrows():
                metadata_value = cleaned_metadata_df.loc[file_id, group_col]
                if metadata_value in group_dict.keys():
                    group_dict[metadata_value].append(file_id)
                else: 
                    group_dict[metadata_value] = [file_id]
            groupings_dict[group_col] = group_dict
    return groupings_dict

def unify_data_sets(data_sets, data_set_identifiers):
    """
    Combine multiple data frames, containing different features of the same samples (!)  

    Args:
        data_sets: set of pandas data frames over the same samples. Each has to have the 
                   same axis 0 (including ordering). 
        data_set_identifiers: set of string identifiers for each df in data sets

    Returns:
        Unified data frame containing all features
    """
    data_list = []
    global_cols = []
    index = data_sets[0].index
    for i, data_set in enumerate(data_sets):
        local_data = data_set.to_numpy()
        local_cols = data_set.columns.to_list()
        data_list.append(local_data)
        globalized_cols = list(map(lambda x: data_set_identifiers[i] + "_" + str(x), local_cols))
        global_cols = global_cols + globalized_cols
    global_data = np.hstack(data_list)
    unified_data_set = pd.DataFrame(data=global_data, index=index, columns=global_cols)
    return unified_data_set

def select_grouping(data_set, groupings, min_group_size):
    """
    select groups of dataset samples by pre-defined groupings passed as a dict of lists.
    lists must contain names that match data set row entries. 

    Args:
        data_set: Data frame which contains the sampley to be grouped. 
        groupings: dictionary defining groups. Keys must be group name, values must be 
                   list of associated sample names (each present in the row idx of the data)
        min_group_size: integer. Filters out all groups with fewer samples.

    Returns:
        list of groupings (data frames containing the samples in the group) and list of masks that
        were used to select said samples.
    """
    group_selections = []
    group_masks = []
    for group in groupings.keys():
        group_sample_idxs = groupings[group]
        group_mask = list(map(lambda x: x in group_sample_idxs), data_set.index)
        if np.sum(group_mask) >= min_group_size: 
            group_selections.append(data_set[group_mask])
            group_masks.append(group_mask)
    return group_selections, group_masks

# NOTE: This process doesnt really make sense. Replace.
def bootstrap_histogram(data, bins, sampling_num = 100):
    bootstrapped_samples = np.random.choice(data, sampling_num)
    binned_index = np.zeros(shape=(len(bins)+1,))
    for sample in bootstrapped_samples:
        if sample > max(bins):
            binned_index[-1] += 1/sampling_num            
        else:
            for i in range(len(bins)):
                if sample <= bins[i]:
                    binned_index[i] += 1/sampling_num
                    break
    return binned_index
        
def get_feature_wise_distribution(groups, group_names, event_sets, sampling_num=100):
    feature_dict = {}
    features = groups[0].columns.to_list()
    for i, group in enumerate(groups):
        group_name = group_names[i]
        local_feature_dict = {}
        for j, feature in enumerate(features):
            event_set = event_sets[j]
            data_array = group.to_numpy()[:,j]
            distribution = bootstrap_histogram(data_array, event_set, sampling_num)
            local_feature_dict[feature] = distribution
        feature_dict[group_name] = local_feature_dict
    return feature_dict

def get_pairwise_feature_divergence(feature_wise_distributions, js_base=2.0):
    data_rows = []
    groups = feature_wise_distributions.keys()
    features = feature_wise_distributions[groups[0]].keys()
    multi_index = []
    for group_one in groups:
        for group_two in groups:
            row = np.zeros((len(features),))
            for i, feature in enumerate(features):
                dist_one = feature_wise_distributions[group_one][feature]
                dist_two = feature_wise_distributions[group_two][feature]
                row[i] += distance.jensenshannon(dist_one, dist_two, base=js_base)
            data_rows.append(row)
            multi_index.append((group_one, group_two))
    data = np.vstack(data_rows)
    index = pd.MultiIndex(multi_index, names=["Group A", "Group B"])
    result_df = pd.DataFrame(data=data, index=index, columns=features)
    return result_df

# NOTE: currently makes a static amount of bins. May want to adjust to hit a certain error instead
def calculate_bins(data_df, num_bins=10):
    data_array = data_df.to_numpy()
    mins = np.min(data_array, axis = 0)
    maxs = np.max(data_array, axis = 0)

    bins_list = []
    for i, maxval in enumerate(maxs):
        value_range = maxval - mins[i]
        bin_size = value_range/num_bins
        bins = [mins[i] + bin_size*n for n in range(num_bins)]
        bins_list.append(bins)

    return bins_list
    

# =============================================================================
# HIGH-LEVEL ANALYSIS FUNCTIONS
# =============================================================================

def grouped_distribution_divergence(dataset, groupings, event_sets, min_group_size=5, sampling_num=100, js_base=2.0):
    """
    Combine multiple data frames, containing different features of the same samples (!)  

    Args:
        dataset: pandas data frame containing the data 
        groupings: dictionary defining groups. Keys must be group name, values must be 
                   list of associated sample names (each present in the row idx of the data)
    """
    group_selections, group_masks = select_grouping(dataset, groupings, min_group_size)
    feature_wise_distributions = get_feature_wise_distribution(group_selections, groupings.keys(), event_sets, sampling_num)
    divergence_df = get_pairwise_feature_divergence(feature_wise_distributions, js_base=js_base)
    return divergence_df

def feature_analysis_pipe(raw_word_src="./data/features/word_histogram_union_raw.csv",
                          pruned_word_src="./data/features/word_histogram_union_pruned.csv",
                          sent_src="./data/features/sentence_lengths_raw.json",
                          metadata_src="./data/metadata.csv",
                          # TODO: find a good set of groups. maybe like 3 ish or so. author, country of origin and institution maybe?
                          groupby=["first_author","primary_category","first_author_country"],
                          visualize=False,
                          save_path=None):
    union_raw_word_df = load_csv(raw_word_src)
    union_pruned_word_df = load_csv(pruned_word_src)
    # NOTE: max_len 50 is chosen here, because it is twice the commonly recommended max sentence lenght of 25
    raw_sentence_df, raw_sentence_dict = load_sentence_json(sent_src, max_len=50)
    metadata_df = load_metadata(metadata_src)

    # get supplementary features
    sentence_stats = get_mean_and_stdev_sent(raw_sentence_df)
    easy_word_ratios = get_easy_words_count(union_raw_word_df)["easy_word_ratio"]
    syllable_counts = get_syllable_counts(union_raw_word_df)
    # NOTE: choice of commonality theshhold was relatively arbitrary
    common_word_counts = get_common_word_df(union_pruned_word_df, commonality_thershhold=0.7)

    # normalize supplementary feature dfs where appropriate
    syllable_ratios_array = normalize_word_rows(syllable_counts.to_numpy())
    syllable_ratios = pd.DataFrame(data=syllable_ratios_array, index=syllable_counts.index, columns=syllable_counts.columns)
    common_word_ratios_array = normalize_word_rows(common_word_counts.to_numpy())
    common_word_ratios = pd.DataFrame(data=common_word_ratios_array, index=common_word_counts.index, columns=common_word_counts.columns)

    # compute groupings
    groupings = get_groupings(metadata_df, groupby) # NOTE: This returns multiple groups collected in a dict of dicts. DO NOT PASS DIRECTLY TO grouped_distribution_divergence
    

    data_set_handles = ["word_freq", "syllable_freq", "easy_freq", "sentence"]
    data_sets = [common_word_ratios, syllable_ratios, easy_word_ratios, sentence_stats]

    dataset = unify_data_sets(data_sets, data_set_handles)

    event_sets = calculate_bins(dataset, num_bins=10)

    result_dfs = {}
    for group_name in groupings.keys():
        distribution_divergence_df = grouped_distribution_divergence(dataset, 
                                                                     groupings=groupings[group_name],
                                                                     event_sets=event_sets,
                                                                     sampling_num=1000)
        result_dfs[group_name] = distribution_divergence_df
    
    if visualize:
        for i, group_name in enumerate(groupings.keys()):
            result_df = result_dfs[i]
            visualize_feature_analysis(result_df, group_name, save_path)

    return result_dfs
        

def analyze_word_histograms(
    csv_path: str,
    output_dir: Optional[str] = None,
    n_components: int = 2,
    perplexity: int = 30
) -> Dict:
    """
    Perform complete analysis on word histogram CSV.

    Args:
        csv_path: Path to word histogram CSV
        output_dir: Directory to save plots (if None, displays interactively)
        n_components: Number of PCA/t-SNE components
        perplexity: t-SNE perplexity parameter

    Returns:
        Dictionary with analysis results
    """
    logger.info(f"Analyzing {csv_path}...")

    # Load and process data
    data_df = load_csv(csv_path)
    author_handles, feature_labels, data = get_np_dataset(data_df)
    data_normalized = normalize_word_rows(data)

    results = {
        'n_documents': len(author_handles),
        'n_features': len(feature_labels),
        'unique_authors': list(set(author_handles))
    }

    # PCA analysis
    logger.info("Running PCA...")
    pca_data, loadings, variance = pca_dim_reduction(data_normalized, n_components)
    results['pca_explained_variance'] = variance.tolist()

    if output_dir:
        from pathlib import Path
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        base_name = Path(csv_path).stem

        plot_dim_reduced_data(
            pca_data, author_handles,
            title=f"PCA - {base_name}",
            save_path=f"{output_dir}/{base_name}_pca.png"
        )
        plot_loadings(
            loadings, feature_labels,
            save_path=f"{output_dir}/{base_name}_loadings"
        )
    else:
        plot_dim_reduced_data(pca_data, author_handles, title="PCA Analysis")
        plot_loadings(loadings, feature_labels)

    # t-SNE analysis (only if enough samples)
    if len(author_handles) > perplexity:
        logger.info("Running t-SNE...")
        tsne_data = tsne_dim_reduction(data_normalized, perplexity=perplexity)

        if output_dir:
            plot_dim_reduced_data(
                tsne_data, author_handles,
                title=f"t-SNE (perplexity={perplexity}) - {base_name}",
                save_path=f"{output_dir}/{base_name}_tsne.png"
            )
        else:
            plot_dim_reduced_data(
                tsne_data, author_handles,
                title=f"t-SNE (perplexity={perplexity})"
            )
    else:
        logger.warning(f"Skipping t-SNE: need more than {perplexity} samples")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "src/data/features/word_histogram_union_pruned.csv"

    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    results = analyze_word_histograms(csv_path, output_dir)
    print(f"Analysis complete: {results}")
