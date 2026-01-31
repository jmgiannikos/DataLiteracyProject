"""
Analysis and visualization module for extracted features.
Provides t-SNE, PCA, and plotting functions for word frequency analysis.

Source: jan-analysis/analysis.py
"""
from statsmodels.nonparametric.kernel_density import KDEMultivariate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from typing import List, Tuple, Optional, Dict, Set
import logging
from scipy.spatial import distance
from plotting import plot_loadings, plot_dim_reduced_data, visualize_df_heatmap, plot_selected_feature_dists, visualize_multiindex_df
from scipy.optimize import LinearConstraint, Bounds, milp
from data_utils import load_csv, load_metadata, load_sentence_json, get_np_dataset, normalize_word_rows, \
    get_mean_and_stdev_sent, get_easy_words_count, get_syllable_counts, get_common_word_df
from collections import defaultdict
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def get_minimal_index(datasets):
    index = list(datasets[0].index)
    for loc_data in datasets[1:]:
        loc_idx = loc_data.index
        missing_idxs = [idx for idx in index if idx not in loc_idx]
        for missing_idx in missing_idxs:
            index.remove(missing_idx)
    return index

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
    index = get_minimal_index(data_sets)
    for i, data_set in enumerate(data_sets):
        local_data = data_set.loc[index].to_numpy()
        local_cols = data_set.columns.to_list()
        data_list.append(local_data)
        globalized_cols = list(map(lambda x: data_set_identifiers[i].format(str(x)), local_cols))
        global_cols = global_cols + globalized_cols
    global_data = np.hstack(data_list)
    unified_data_set = pd.DataFrame(data=global_data, index=index, columns=global_cols)
    return unified_data_set

def select_grouping(data_set, groupings, crossval_split=0):
    """
    select groups of dataset samples by pre-defined groupings passed as a dict of lists.
    lists must contain names that match data set row entries. 

    Args:
        data_set: Data frame which contains the sampley to be grouped. 
        groupings: dictionary defining groups. Keys must be group name, values must be 
                   list of associated sample names (each present in the row idx of the data)
        min_group_size: integer. Filters out all groups with fewer samples.

    Returns:
        list of groupings (data frames containing the samples in the group) if crossval split is <=1.
        otherwise returns dict of crossval splits dictionary containing dictionaries with test and train
        grouping dfs.
    """
    group_selections = {}
    for group in groupings.keys():
        group_sample_idxs = groupings[group]
        group_mask = list(map(lambda x: x in group_sample_idxs, data_set.index))
        group_selections[group] = data_set[group_mask]

    if crossval_split < 1:
        crossval_split = 1

    split_sizes = [max([int(group_selections[group_name].shape[0]/crossval_split), 1]) for group_name in group_selections.keys()]
    lastsplit_size = [group_selections[group_name].shape[0] - split_sizes[idx] * (crossval_split-1) for idx, group_name in enumerate(group_selections.keys())]
    splits = {}
    for split_id in range(crossval_split):
        if split_id != crossval_split-1:
            holdout_idxs = [range(split_id*split_sizes[i], (split_id+1)*split_sizes[i]) for i in range(len(split_sizes))]
        else:
            holdout_idx = [range(split_id*split_sizes[i], split_id*split_sizes[i]+lastsplit_size[i]) for i in range(len(split_sizes))]
        test_sets = {}
        train_sets = {}
        for group_idx, holdout_idx in enumerate(holdout_idxs):
            group_name = list(groupings.keys())[group_idx]
            holdout_idx_list = [group_selections[group_name].index[i] for i in holdout_idx]
            selection_mask = group_selections[group_name].index.isin(holdout_idx_list)
            test_sets[group_name] = group_selections[group_name][selection_mask]
            train_sets[group_name] = group_selections[group_name][~selection_mask]
        splits[split_id] = {"train": train_sets, "test": test_sets}
    return splits

def sort_to_bins(data, bins):
    binned_index = np.zeros(shape=(len(bins)+1,))
    for sample in data:
        if sample > max(bins):
            binned_index[-1] += 1/data.shape[0]            
        else:
            for i in range(len(bins)):
                if sample <= bins[i]:
                    binned_index[i] += 1/data.shape[0]
                    break
    return binned_index

def kde_discrete(estimator, bins):
    if estimator is None:
        p_bins = np.zeros((len(bins)+1,))
        p_bins[0] = 1
        return p_bins
    else:
        cdf_bins = estimator.cdf(bins) # assume number in bins represent respective upper bounds
        p_bins = [cdf_bins[i]-cdf_bins[i-1] for i in range(1,len(cdf_bins))]
        p_bins = [cdf_bins[0]] + p_bins + [1-cdf_bins[-1]]
        return p_bins

def get_kde_estimators(data):
    if np.max(data) <= 0:
        return None
    else:
        reshaped_data = data.reshape(-1, 1)
        kde = KDEMultivariate(reshaped_data,var_type="c", bw="cv_ml")
        return kde
    
def get_global_range(groups, feature):
    data_arrays = []
    for group_name in groups.keys():
        group = groups[group_name]
        data_arrays.append(group[feature].to_numpy())
    global_data_array = np.hstack(data_arrays)
    min_val = np.min(global_data_array)
    max_val = np.max(global_data_array)
    val_range = max_val - min_val
    return min_val, max_val, val_range

def get_pairwise_divergence_df(groups, num_bins):
    first_group = list(groups.keys())[0]
    features = groups[first_group].columns.to_list()
    feature_divergence_vecs = {}
    for feature in features:
        distribution_estimators = {}
        for group_name in groups.keys():
            group = groups[group_name]
            data_array = group[feature].to_numpy()
            distribution_estimators[group_name] = get_kde_estimators(data_array)

        min_value, max_value, value_range = get_global_range(groups, feature)
        bins = np.arange(min_value, max_value, step=value_range/num_bins)
        pairwise_dist_vec = get_pairwise_feature_divergence(groups, distribution_estimators, bins)
        feature_divergence_vecs[feature] = pairwise_dist_vec

    row_idxs = feature_divergence_vecs[features[0]].keys()
    rows = []
    index = pd.MultiIndex.from_tuples(row_idxs, names=("Author A", "Author B"))
    for row_idxa, row_idxb in row_idxs:
        row = np.array([feature_divergence_vecs[feature][(row_idxa, row_idxb)] for feature in features])
        rows.append(row)
    data_array = np.vstack(rows)
    result_df = pd.DataFrame(data=data_array, index=index, columns=features)
    return result_df

def get_pairwise_feature_divergence(groups, distribution_estimators, bins):
    group_names = groups.keys()
    evaluated_groups = []
    distances = {}
    for group_name_a in group_names:
        evaluated_groups.append(group_name_a)
        for group_name_b in group_names:
            if group_name_b not in evaluated_groups:
                distribution_estimator_a = distribution_estimators[group_name_a]
                distribution_estimator_b = distribution_estimators[group_name_b]
                distribution_a = kde_discrete(distribution_estimator_a, bins)
                distribution_b = kde_discrete(distribution_estimator_b, bins)
                dist = distance.jensenshannon(distribution_a, distribution_b, base=2)
                distances[(group_name_a, group_name_b)] = dist
    return distances

# generates bins as list of upper bounds
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
    
def remove_redundant_rows(pairwise_divergence_df):
    pruned_idx = []
    for index_tup in pairwise_divergence_df.index:
        index_set = set(index_tup)
        if index_tup[0] != index_tup[1] and index_set not in pruned_idx:
            pruned_idx.append(index_set)
    pruned_idx = [tuple(index_set) for index_set in pruned_idx]
    return pairwise_divergence_df.loc[pruned_idx]

# averages prediction stack over groups. The prediction lies on axis y while the ground truth class lies on axis x of the returned df
def average_over_row_group(dataframes):
    if isinstance(dataframes, dict):
        df_list = [dataframes[key] for key in dataframes.keys()]
        unified_df = pd.concat(df_list, axis=0)
    else:
        unified_df = dataframes
    row_groups = list(unified_df.columns) # NOTE: this code was changed from the original list(set(unified_df.index)) to current state so that the columns align with the rows (ordering)
    result_rows = []
    for group in row_groups:
        group_mask = unified_df.index.to_numpy() == group
        group_selection = unified_df.to_numpy()[group_mask]
        group_row = np.mean(group_selection, axis=0)
        result_rows.append(group_row)
    result_array = np.vstack(result_rows)
    result_df = pd.DataFrame(data=result_array, index=row_groups, columns=unified_df.columns)
    return result_df

def get_confusion_mats(dataframes):
    if isinstance(dataframes, dict):
        df_list = [dataframes[key] for key in dataframes.keys()]
        unified_df = pd.concat(df_list, axis=0)
    else:
        unified_df = dataframes
    pred_labels = unified_df.columns
    row_groups = list(unified_df.columns)
    result_rows = []
    for group in row_groups:
        group_mask = unified_df.index.to_numpy() == group
        group_selection = unified_df.to_numpy()[group_mask]
        row_maxes = group_selection.max(axis=1).reshape(-1, 1)
        group_row = np.where(group_selection == row_maxes, 1, 0)
        group_row = np.sum(group_row, axis=0)
        result_rows.append(group_row)
    result_array = np.vstack(result_rows)
    result_df = pd.DataFrame(data=result_array, index=row_groups, columns=pred_labels)
    return result_df

# =============================================================================
# HIGH-LEVEL ANALYSIS FUNCTIONS
# =============================================================================

def select_features(pairwise_divergence_df, select_num=2):
    feature_selection_vector = greedy_feature_select(pairwise_divergence_df.to_numpy(), select_num=select_num)
    selected_features = np.array(pairwise_divergence_df.columns)[feature_selection_vector] # binary masking operation
    selected_df = pairwise_divergence_df[selected_features]
    return selected_df, selected_features

def greedy_feature_select(divergence_mat, select_num):
    if select_num >= divergence_mat.shape[1]:
        return np.ones((divergence_mat.shape[1],)) == 1
    selection_vector = np.zeros((divergence_mat.shape[1],))
    selection_order = []
    weight_vec = np.ones((divergence_mat.shape[0],))
    for _ in range(select_num):
        # greedily choose feature with highest weighted divergence
        weighted_result_vec = np.max((divergence_mat*(1-selection_vector)).T * weight_vec, axis=1)
        max_idx = np.argmax(weighted_result_vec)
        assert selection_vector[max_idx] == 0
        selection_vector[max_idx] = 1
        selection_order.append(max_idx)
        weight_vec = 1 - np.max(divergence_mat * selection_vector, axis=1)
    return selection_order # selection_vector == 1

def grouped_distribution_divergence(group_selections, num_bins=100):
    """
    Combine multiple data frames, containing different features of the same samples (!)  

    Args:
        dataset: pandas data frame containing the data 
        groupings: dictionary defining groups. Keys must be group name, values must be 
                   list of associated sample names (each present in the row idx of the data)
        crossval_split: number of splits to perform crossval over

    Returns:
        if crossval_split <= 1 returns feature wise divergence df. otherwise returns crossval split dict of dicts,
        where each split has a divergence df, a test set and a train set.
    """
    group_names = group_selections[0]["train"].keys()
    results = {}
    for split_id in group_selections.keys():
        train_selection = group_selections[split_id]["train"]
        test_selection = group_selections[split_id]["test"]

        # some sanity checks
        assert all([group_name in group_names for group_name in train_selection.keys()])
        assert len(group_names) == len(train_selection.keys())

        pairwise_divergence_df = get_pairwise_divergence_df(train_selection, num_bins=num_bins)
        results[split_id] = {"div_df": pairwise_divergence_df, "test": test_selection, "train": train_selection, "group_names": group_names}
    return results
    
def assemble_dataset(raw_word_src="./data/features/word_histogram_union_raw.csv",
                     pruned_word_src="./data/features/word_histogram_union_pruned.csv",
                     sent_src="./data/features/sentence_lengths_raw.json",
                     metadata_src="./data/metadata.csv",
                     commonality_threshhold=0.8,
                     groupby=["first_author","first_author_institution","first_author_country"],
                     crossval_splits=1):
    metadata_df = load_metadata(metadata_src)
    union_raw_word_df = load_csv(raw_word_src)
    union_pruned_word_df = load_csv(pruned_word_src)
    # NOTE: max_len 50 is chosen here, because it is twice the commonly recommended max sentence lenght of 25
    raw_sentence_df, raw_sentence_dict = load_sentence_json(sent_src, max_len=45)

    # get supplementary features
    sentence_stats = get_mean_and_stdev_sent(raw_sentence_df)
    easy_word_ratios = get_easy_words_count(union_raw_word_df)[["easy_word_ratio"]]
    syllable_counts = get_syllable_counts(union_raw_word_df)
    # NOTE: choice of commonality theshhold was relatively arbitrary
    common_word_counts = get_common_word_df(union_pruned_word_df, commonality_thershhold=commonality_threshhold, cutoff=-1)

    syllable_ratios_array = normalize_word_rows(syllable_counts.to_numpy())
    syllable_ratios = pd.DataFrame(data=syllable_ratios_array, index=syllable_counts.index, columns=syllable_counts.columns)
    common_word_ratios_array = normalize_word_rows(common_word_counts.to_numpy())
    common_word_ratios = pd.DataFrame(data=common_word_ratios_array, index=common_word_counts.index, columns=common_word_counts.columns)
    data_set_handles = ["'{}' freq.", "{}-syllable freq.", "easy word ratio", "sentence {}"]
    data_sets = [common_word_ratios, syllable_ratios, easy_word_ratios, sentence_stats]
    dataset = unify_data_sets(data_sets, data_set_handles)
    
    group_axes = get_groupings(metadata_df, groupby)
    group_selections = {}
    for group_axis in group_axes.keys():
        axis_wise_groupings = group_axes[group_axis]
        group_selections[group_axis] = select_grouping(dataset, axis_wise_groupings, crossval_split=crossval_splits)

    return group_selections

def predict(selected_features, train_samples, test_samples):
    train_dict = {}
    test_dict = {}
    for group_name in train_samples.keys():
        train_dict[group_name] = train_samples[group_name][selected_features]
        test_dict[group_name] = test_samples[group_name][selected_features].to_numpy()
    group_predictor = Group_Predictor_statsmodels()
    group_predictor.fit(train_dict)
    group_prediction = group_predictor.predict(test_dict)
    return group_prediction

def merge_div_dfs(div_dfs, select_num):
    if len(div_dfs.keys()) == 1:
        return div_dfs[0], pd.DataFrame(data=np.zeros_like(div_dfs[0]), columns=div_dfs[0].columns, index=div_dfs[0].index)
    else:
        value_arrays = [np.expand_dims(div_dfs[0].to_numpy(), 0)]
        columns = div_dfs[0].columns
        index = div_dfs[0].index
        for split_id in div_dfs.keys()[1:]:
            div_df = div_dfs[split_id]
            assert all(columns == div_df.colums)
            assert all(index == div_df.index)
            value_arrays.append(np.expand_dims(div_df.to_numpy(),0))
        stacked_values = np.append(value_arrays[0], value_arrays[1:], axis=0)
        mean_values = np.mean(stacked_values, axis=0, keepdims=False)
        stdev_values = np.sqrt(np.var(stacked_values, axis=0, keepdims=False))
        mean_df = pd.DataFrame(data=mean_values, index=index, columns=columns)
        stdev_values = pd.DataFrame(data=stdev_values, index=index, columns=columns)
        selected_features_df, selected_features = select_features(mean_df, select_num=select_num)
        return selected_features_df, stdev_values[selected_features]        

def get_selected_features_ratios(selected_features_dict):
    splits = len(selected_features_dict.keys())
    selected_feature_ratios = {}
    for split_id in selected_features_dict.keys():
        selected_features = selected_features_dict[split_id]
        for selected_feature in selected_features:
            if selected_feature in selected_feature_ratios.keys():
                selected_feature_ratios[selected_feature] += 1/splits
            else:
                selected_feature_ratios[selected_feature] = 1/splits
    return selected_feature_ratios

def prediction_pipe(raw_word_src="./data/features/word_histogram_union_raw_category.csv",
                    pruned_word_src="./data/features/word_histogram_union_pruned_category.csv",
                    sent_src="./data/features/sentence_lengths_raw_category.json",
                    metadata_src="./data/top8-authors-category.csv",
                    groupby=["first_author"],
                    crossval_splits=5,
                    select_num=2,
                    commonality_threshhold=0.95,
                    num_feat_select_bins=100):
    
    axis_group_selections = assemble_dataset(raw_word_src=raw_word_src, 
                                        pruned_word_src=pruned_word_src,
                                        sent_src=sent_src,
                                        metadata_src=metadata_src,
                                        commonality_threshhold=commonality_threshhold,
                                        groupby=groupby,
                                        crossval_splits=crossval_splits)
    predictions_dict = {}
    performances_dict = {}
    selected_feature_ratios_dict = {}
    mean_selected_feature_df_dict = {}
    stdev_selected_feature_df_dict = {}
    for group_axis in axis_group_selections.keys():
        group_selections = axis_group_selections[group_axis]
        result_dict = grouped_distribution_divergence(group_selections=group_selections, num_bins=num_feat_select_bins)

        selected_features_dict = {}
        div_dfs = {}
        split_predictions = {}
        for split_id in result_dict.keys():
            div_df = result_dict[split_id]["div_df"]
            test_set = result_dict[split_id]["test"]
            train_set = result_dict[split_id]["train"]
            selected_feature_df, selected_features = select_features(div_df, select_num=select_num)
            div_dfs[split_id] = div_df
            selected_features_dict[split_id] = selected_features
            prediction_result = predict(selected_features=selected_features, test_samples=test_set, train_samples=train_set)
            split_predictions[split_id] = prediction_result

        avg_performance = average_over_row_group(split_predictions)
        conf_mats = get_confusion_mats(split_predictions)
        selected_feature_ratios = get_selected_features_ratios(selected_features_dict)
        mean_selected_feature_df, stdev_selected_feature_df = merge_div_dfs(div_dfs, select_num=select_num)

        predictions_dict[group_axis] = avg_performance
        performances_dict[group_axis] = conf_mats
        selected_feature_ratios_dict[group_axis] = selected_feature_ratios
        mean_selected_feature_df_dict[group_axis] = mean_selected_feature_df
        stdev_selected_feature_df_dict[group_axis] = stdev_selected_feature_df

    return predictions_dict, performances_dict, selected_feature_ratios_dict, mean_selected_feature_df_dict, stdev_selected_feature_df_dict

def analyze_word_histograms(
    csv_path: str,
    output_dir: Optional[str] = None,
    n_components: int = 2,
    perplexity: int = 30,
    metadata_path: str = "data/metadata.csv"
) -> Dict:
    """
    Perform complete analysis on word histogram CSV.

    Args:
        csv_path: Path to word histogram CSV
        output_dir: Directory to save plots (if None, displays interactively)
        n_components: Number of PCA/t-SNE components
        perplexity: t-SNE perplexity parameter
        metadata_path: Path to metadata CSV for author lookup

    Returns:
        Dictionary with analysis results
    """
    from pathlib import Path
    logger.info(f"Analyzing {csv_path}...")

    # Load metadata if available
    metadata_df = None
    if Path(metadata_path).exists():
        try:
            metadata_df = load_metadata(metadata_path)
            logger.info(f"Loaded metadata from {metadata_path} for author lookup")
        except Exception as e:
            logger.warning(f"Failed to load metadata from {metadata_path}: {e}")

    # Load and process data
    data_df = load_csv(csv_path)
    author_handles, feature_labels, data = get_np_dataset(data_df, metadata_df)
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

class Group_Predictor_statsmodels:
    def __init__(self, bandwidth="cv_ml", kernel="gaussian", assume_equal_p_groups=True, eps=np.finfo(np.float64).eps):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.assume_equal_p_groups = assume_equal_p_groups
        self.eps = eps

    def fit(self, sample_df_dict):
        self.group_names = list(sample_df_dict.keys())
        self.feature_names = list(sample_df_dict[self.group_names[0]].columns) # assume all dfs have the same cols
        
        group_labels = []
        sample_sets = []
        for key in self.group_names:
            sample_set = sample_df_dict[key].to_numpy()
            sample_sets.append(sample_set)
            group_labels = group_labels + [self.group_names.index(key)]*sample_set.shape[0]
        
        all_samples = np.vstack(sample_sets)
        total_samples = all_samples.shape[0]
        total_features =  all_samples.shape[1]
        author_labelled_samples = np.hstack((np.expand_dims(np.array(group_labels).T, 1), all_samples))

        if not self.assume_equal_p_groups:
            self.p_groups = [sample_set.shape[0]/total_samples for sample_set in sample_sets]
        else:
            self.p_groups = [1/len(self.group_names)]*len(self.group_names)
        
        self.p_joint = KDEMultivariate(author_labelled_samples, var_type="u"+"c"*total_features, bw="cv_ml")
        # NOTE: if the estimator sets this bw to 1 (which it sometimes does), the aitchison-aitken kernels return zero for all samples that have the same category as the sample around which the kernel is centered. This causes the model to effectively ignore entire groupings of kernels and results in inverted performance
        # to sidestep this issue, we set bw to 0.5, which should be equivalent to weighting by the incoming samples
        self.p_joint.bw[0] = 0.5 

    def predict(self, sample_array_dict):
        sample_ids = []
        for key in sample_array_dict.keys():
            sample_ids = sample_ids + [key]*sample_array_dict[key].shape[0]
            
        samples = np.vstack([sample_array_dict[key] for key in sample_array_dict.keys()])
        ps_group_given_sample = []
        for group in self.group_names:
            group_idx = self.group_names.index(group)
            group_idx_arr = np.ones((samples.shape[0], 1)) * group_idx
            full_samples = np.hstack((group_idx_arr, samples))
            likelihood_group_given_sample = self.p_joint.pdf(full_samples)
            ps_group_given_sample.append(likelihood_group_given_sample.T)
        p_feats = np.sum(np.vstack(ps_group_given_sample), axis=0) # marginalize authors out
        p_feats[p_feats == 0] = self.eps # set zero values to eps for numerical stability
        result_probs = []
        for ps in ps_group_given_sample:
            result_probs.append(ps/p_feats)
        author_probs = np.vstack(result_probs).T
        author_prob_df = pd.DataFrame(data=author_probs, index=sample_ids, columns=self.group_names)
        return author_prob_df

def main():
    raw_word_src="./data/features/word_histogram_union_raw_category.csv"
    pruned_word_src="./data/features/word_histogram_union_pruned_category.csv"
    sent_src="./data/features/sentence_lengths_raw_category.json"
    metadata_src="./data/top8-authors-category.csv"
    select_nums = [4]
    for select_num in select_nums:
        predictions, performances, feature_selection_ratios, feature_selection_dfs, feature_selection_stdev_dfs = prediction_pipe(raw_word_src=raw_word_src, 
                        pruned_word_src=pruned_word_src, 
                        sent_src=sent_src, 
                        metadata_src=metadata_src, 
                        select_num=select_num,
                        commonality_threshhold=0.9,
                        num_feat_select_bins=100,
                        groupby=["first_author"])
        for key in predictions.keys():
            prediction = predictions[key]
            performance = performances[key]
            feature_selection_ratio = feature_selection_ratios[key]
            feature_selection_df = feature_selection_dfs[key]
            feature_selection_stdev_df = feature_selection_stdev_dfs[key]
            print(feature_selection_ratio)
            visualize_multiindex_df(feature_selection_df, key + "_feature_analysis", f"./feat_analysis_features_{str(select_num).replace(".",",")}.png", False)
            visualize_multiindex_df(feature_selection_stdev_df, key + "_feature_analysis_stdev", f"./feat_analysis_stdev_features_{str(select_num).replace(".",",")}.png", False)
            visualize_df_heatmap(prediction, key + "_prediction_performance", f"./pred_performance_features_{str(select_num).replace(".",",")}.png", True, True)
            visualize_df_heatmap(performance, key + "_prediction_conf_mat", f"./pred_conf_mat_features_{str(select_num).replace(".",",")}.png", True, True)

if __name__ == "__main__":
    main()
