import numpy as np
import pandas
import pandas as pd
from typing import List, Tuple, Optional, Dict, Set
import logging
import json
import textstat

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
                row[value - 1] += 1

        rows.append(row)

    np_data = np.vstack(rows)
    sentence_df = pd.DataFrame(data=np_data, index=index, columns=range(1, max_val + 1))
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


def merge_authors(metadata_df, author_name, alias):
    """
    Edits the metatdata dataframe:
    In case one author has multiple aliases, merge the data for this author, so all papers are under the name given by author_name
    Example: 'Florentin Millour' and 'F. Millour' are falsely listed as one author
    Also works with two lists of same length, e.g.
    merge_authors(['Florentin Millour', 'James Leftley'], ['F. Millour', 'J. Leftley'])

    Args:
        metadata_df: Dataframe with Metadata
        author_name: Name of the author
        alias: Alias (Will be replaces)

    Returns:
        edited metadata
    """
    new_metadata_df = metadata_df.replace(alias, author_name)
    return new_metadata_df


# =============================================================================
# SUPPLEMENTARY FEATURE EXTRACTION
# =============================================================================

def get_common_words(word_hist_dataframe, commonality_thershhold=0.5, cutoff=-1):
    common_words = []
    commonality_ratios = []
    recorded_documents, _ = word_hist_dataframe.shape
    for word in word_hist_dataframe.columns:
        counts = word_hist_dataframe[word]
        commonality_ratio = sum([1 if count > 0 else 0 for count in counts]) / recorded_documents
        if commonality_ratio >= commonality_thershhold:
            common_words.append(word)
            commonality_ratios.append(commonality_ratio)
    zipped_common_words = list(zip(commonality_ratios, common_words))
    zipped_common_words.sort(reverse=True, key=lambda x: x[0])
    sorted_common_words = [x[1] for x in zipped_common_words]
    if cutoff > 0:
        return sorted_common_words[:cutoff]
    else:
        return common_words


def get_common_word_df(words_df, commonality_thershhold=0.5, cutoff=-1):
    common_words = get_common_words(words_df, commonality_thershhold, cutoff=cutoff)
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
def get_syllable_counts(words_df):
    """
    Takes a non-normalized word histogram dataframe and computes the syllable count distribution

    Args:
        words_df: a dataframe containing word counts. Row index must be document identifiers, col
                  must be words.

    Returns:
        a dataframe with syllable counts as col index and document identifiers as row idndex
    """
    # Compute syllable count for each word
    word_list = words_df.columns.tolist()
    syllable_counts = np.array([textstat.syllable_count(word) for word in word_list])
    # Determine maximum syllable count
    max_syllable_count = np.max(syllable_counts)
    # prepare result array
    result_arr = np.zeros((words_df.shape[0], max_syllable_count))
    # convert dataframe to array
    word_arr = words_df.to_numpy()

    for s in range(1, max_syllable_count + 1):
        s_mask = syllable_counts == s
        result_arr[:, s-1] = word_arr[:, s_mask].sum(axis=1)

    syllable_count_df = pd.DataFrame(data=result_arr, index=words_df.index, columns=range(1, max_syllable_count + 1))
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
    is_easy_mask = [0] * len(words_df.columns)
    for col_idx, word in enumerate(words_df.columns):
        if textstat.is_easy_word(word):
            is_easy_mask[col_idx] = 1
    is_easy_mask = np.array(is_easy_mask)

    rows = []
    row_index = []
    for row_id, row in words_df.iterrows():
        easy_word_count = np.inner(row.to_numpy(), is_easy_mask)
        easy_word_ratio = easy_word_count / np.sum(row.to_numpy())
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
    cols_np = np.hstack((np.expand_dims(monosyllabic_count_array, 1), np.expand_dims(monosyllabic_ratio, 1)))
    monosyllabic_count_df = pd.DataFrame(data=cols_np, index=word_df.index, columns=["monosyl_count", "monosyl_ratio"])
    return monosyllabic_count_df

def get_paper_ids_for_author(author: str, metadata_df: pd.DataFrame):
    """
    Retrieves all Papers where given author is the first author
    Args:
        metadata_df: dataframe with metadata
        author: author name

    Returns:
        List with paper ids
    """
    paper_ids = list(metadata_df.query(f"first_author=='{author}'")["arxiv_id"])
    return paper_ids

def get_word_count_for_paper(paper_id, union_raw_df, words='all'):
    """
        Gets the total word count for a list of words, if 'all' it will get the total word count of this paper
    """
    word_count = 0
    if words == 'all':
        word_row = union_raw_df[union_raw_df['Data Name'] == paper_id].values.flatten()
        word_count += int(np.sum(word_row[1:]))
    else:
        for word in words:
            word_count += int(union_raw_df[union_raw_df['Data Name'] == paper_id][word].values[0])
    return word_count

def get_word_count_for_author(author, metadata_df, union_raw_df, words='all'):
    """
        Gets the total word count for a list of words, if 'all' it will get the total word count of this author
    """
    word_count = 0
    paper_ids = get_paper_ids_for_author(author, metadata_df)
    if len(paper_ids) == 0:
        print(f'Error: could not find papers for {author}')
    else:
        for paper in paper_ids:
            word_count += get_word_count_for_paper(paper,union_raw_df, words=words)
    return int(word_count)

def total_words_by_author(list_of_authors, metadata_df, union_raw_df):
    author_total_words_dict = {}
    for author in list_of_authors:
        author_total_words_dict[author] = get_word_count_for_author(author, metadata_df, union_raw_df)
    return author_total_words_dict

def remove_paper_from_dfs(paper_id, metadata_df, union_raw, union_pruned, inter_raw, inter_pruned):
    metadata_i = metadata_df[metadata_df.arxiv_id == paper_id].index
    metadata_df = metadata_df.drop(metadata_i, axis=0)
    i = union_raw[union_raw["Data Name"] == paper_id].index
    union_raw, union_pruned, inter_raw, inter_pruned = map(lambda df: df.drop(i, axis=0), [union_raw, union_pruned, inter_raw, inter_pruned])
    return metadata_df, union_raw, union_pruned, inter_raw, inter_pruned

def remove_authors(author_list, metadata_df, union_raw, union_pruned, inter_raw, inter_pruned):
    new_m, new_u_r, new_u_p, new_i_r, new_i_p = metadata_df, union_raw, union_pruned, inter_raw, inter_pruned
    a = 0
    p = 0
    for author in author_list:
        a += 1
        paper_list = get_paper_ids_for_author(author, metadata_df)
        for paper_id in paper_list:
            p += 1
            new_m, new_u_r, new_u_p, new_i_r, new_i_p = remove_paper_from_dfs(paper_id, new_m, new_u_r, new_u_p, new_i_r, new_i_p)
    print(f'removed a total of {p} papers by {a} authors')
    return new_m, new_u_r, new_u_p, new_i_r, new_i_p

def remove_papers_if_empty(metadata_df, union_raw, union_pruned, inter_raw, inter_pruned):
    new_m, new_u_r, new_u_p, new_i_r, new_i_p = metadata_df, union_raw, union_pruned, inter_raw, inter_pruned
    n = 0
    for paper_id in metadata_df["arxiv_id"].tolist():
        if get_word_count_for_paper(paper_id, union_raw, words='all') == 0:
            n += 1
            new_m, new_u_r, new_u_p, new_i_r, new_i_p = remove_paper_from_dfs(paper_id, new_m, new_u_r, new_u_p, new_i_r, new_i_p)
    print(f'removed {n} empty papers')
    return new_m, new_u_r, new_u_p, new_i_r, new_i_p

def remove_papers_with_sen_length(df, minlen, maxlen):
    new_df = df[df.mean_sentence_length < maxlen]
    new_df = new_df[new_df.mean_sentence_length > minlen]
    return new_df