"""
Analysis and visualization module for extracted features.
Provides t-SNE, PCA, and plotting functions for word frequency analysis.

Source: jan-analysis/analysis.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from typing import List, Tuple, Optional, Dict, Set
import logging
from scipy.spatial import distance
from plotting import plot_loadings, plot_dim_reduced_data, visualize_df_heatmap
from scipy.optimize import LinearConstraint, Bounds, milp
from data_utils import load_csv, load_metadata, load_sentence_json, get_np_dataset, normalize_word_rows, \
    get_mean_and_stdev_sent, get_easy_words_count, get_syllable_counts, get_common_word_df
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
        globalized_cols = list(map(lambda x: data_set_identifiers[i] + "_" + str(x), local_cols))
        global_cols = global_cols + globalized_cols
    global_data = np.hstack(data_list)
    unified_data_set = pd.DataFrame(data=global_data, index=index, columns=global_cols)
    return unified_data_set

def select_grouping(data_set, groupings, min_group_size, crossval_split=0):
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
    group_selections = []
    group_masks = []
    retained_group_names = []
    x = 0
    for group in groupings.keys():
        group_sample_idxs = groupings[group]
        group_mask = list(map(lambda x: x in group_sample_idxs, data_set.index))
        if np.sum(group_mask) >= min_group_size: 
            group_selections.append(data_set[group_mask])
            group_masks.append(group_mask)
            retained_group_names.append(group)
        else:
            x+=1
    if x == len(list(groupings.keys())):
        print(f'Minimum Group Size too large ({min_group_size}), could not select any group')

    if crossval_split <= 1:
        return group_selections, retained_group_names
    else:
        # TODO: crossval splitting isnt super clean. Not every sample may be included in a split due to rounding (currently strictly rounding down)
        split_sizes = [max([int(group_selection.shape[0]/crossval_split), 1]) for group_selection in group_selections]
        splits = {}
        for split_id in range(crossval_split):
            holdout_idxs = [range(split_id*split_sizes[i], (split_id+1)*split_sizes[i]) for i in range(len(split_sizes))]
            test_sets = []
            train_sets = []
            for group_idx, holdout_idx in enumerate(holdout_idxs):
                holdout_idx_list = [group_selections[group_idx].index[i] for i in holdout_idx]
                selection_mask = group_selections[group_idx].index.isin(holdout_idx_list)
                test_sets.append(group_selections[group_idx][selection_mask])
                train_sets.append(group_selections[group_idx][~selection_mask])
            splits[split_id] = {"train": train_sets, "test": test_sets}
        return splits, retained_group_names



# NOTE: This process doesnt really make sense. Replace.
def bootstrap_histogram(data, bins, i, j, sampling_num = 100):
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
    if j == 1:
        plt.stairs(binned_index)
        plt.stairs(data)
        plt.savefig(f"./data/analysis/bootstrap_{i}_{j}.png", dpi=150, bbox_inches='tight')
    return binned_index

def plot_data_vs_samples(data, samples, bins, i, j):
    plt.clf()
    plt.hist(data, bins, label='Data', density=True, fill=False, edgecolor='red', linewidth=3)
    plt.hist(samples, bins, label='Sampling', density=True, fill=False, edgecolor='blue')
    plt.legend()
    plt.savefig(f"./data/analysis/kde_discrete_dist_{i}_{j}.png", dpi=150, bbox_inches='tight')

def convert_bins(bins, data):
    # converting bins into being matplotlib.hist() compatible
    # not optimal implementation but does the job
    end_bin = float(max(np.max(data), np.max(bins) * 1.5))
    if np.sum(data) == 0:
        start_bin = np.min(0.5 * bins[bins != 0])
    else:
        start_bin = float(min(np.min(data[data!=0]), np.min(0.5 * bins[bins!=0])))
    converted_bins = np.concatenate([[bins[0]], [start_bin], bins[1:], [end_bin]])
    return converted_bins

def kde_discrete(data, bins, i, j, bandwidth_scale=0.5, sampling_num=100):
    bandwidth = bandwidth_scale * 0.5 * (bins[1] - bins[0])
    reshaped_data = data.reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(reshaped_data)
    sampled_data = np.array(kde.sample(sampling_num))
    converted_bins = convert_bins(bins, data)
    # Removing values < 0
    corrected_samples = sampled_data[sampled_data >= 0]
    # This can be uncommented if one wants to visually compare data and sample results
    #if j in range(0,20,2):
    #    plot_data_vs_samples(data, corrected_samples, converted_bins, i, j)
    kde_discrete_dist, bin_edges, patches = plt.hist(corrected_samples, converted_bins, density=True)
    return kde_discrete_dist

        
def get_feature_wise_distribution(groups, group_names, event_sets, sampling_num=100):
    feature_dict = {}
    features = groups[0].columns.to_list()
    for i, group in enumerate(groups):
        group_name = group_names[i]
        local_feature_dict = {}
        for j, feature in enumerate(features):
            event_set = event_sets[j]
            data_array = group.to_numpy()[:,j]
            #distribution_bootstrap = bootstrap_histogram(data_array, event_set, i, j, sampling_num=sampling_num)
            distribution_kde = kde_discrete(data_array, event_set, i, j, sampling_num=sampling_num)
            local_feature_dict[feature] = distribution_kde
        feature_dict[group_name] = local_feature_dict
    return feature_dict

def get_pairwise_feature_divergence(feature_wise_distributions, js_base=2.0):
    data_rows = []
    groups = list(feature_wise_distributions.keys())
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
    index = pd.MultiIndex.from_tuples(multi_index, names=("Group A", "Group B"))
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
    
def remove_redundant_rows(pairwise_divergence_df):
    pruned_idx = []
    for index_tup in pairwise_divergence_df.index:
        index_set = set(index_tup)
        if index_tup[0] != index_tup[1] and index_set not in pruned_idx:
            pruned_idx.append(index_set)
    pruned_idx = [tuple(index_set) for index_set in pruned_idx]
    return pairwise_divergence_df.loc[pruned_idx]

def average_over_row_group(dataframes):
    if isinstance(dataframes, list):
        unified_df = pd.concat(dataframes, axis=0)
    else:
        unified_df = dataframes
    row_groups = list(set(unified_df.index))
    result_rows = []
    for group in row_groups:
        group_mask = unified_df.index.to_numpy() == group
        group_selection = unified_df.to_numpy()[group_mask]
        group_row = np.mean(group_selection, axis=0)
        result_rows.append(group_row)
    result_array = np.vstack(result_rows)
    result_df = pd.DataFrame(data=result_array, index=row_groups, columns=unified_df.columns)
    return result_df


# =============================================================================
# HIGH-LEVEL ANALYSIS FUNCTIONS
# =============================================================================

def select_features(pairwise_divergence_df, lower_divergence_bound=0, degrade_step=0.01):
    pruned_df = remove_redundant_rows(pairwise_divergence_df)
    feature_selection_vector = feature_selection_optimizer(pruned_df.to_numpy(), lower_divergence_bound, degrade_step)
    selected_features = np.array(pairwise_divergence_df.columns)[feature_selection_vector] # binary masking operation
    selected_df = pairwise_divergence_df[selected_features]
    return selected_df, selected_features

def feature_selection_optimizer(pairwise_divergences, lower_divergence_bound=0, degrade_step=-1):
    num_features = np.shape(pairwise_divergences)[1]
    num_constraints = np.shape(pairwise_divergences)[0]

    # constrain solution to be binary vector
    bool_bound = Bounds(lb=np.zeros((num_features,)), ub=np.ones((num_features,)))
    integrality = np.ones((num_features,))

    # setup constraints
    if not isinstance(lower_divergence_bound, list):
        lower_divergence_bound = [lower_divergence_bound]*num_constraints
    constraint = LinearConstraint(A=pairwise_divergences, lb=lower_divergence_bound)
    
    # setup sum contraint
    c = np.ones((num_features,))/num_features # normalize so we know the max, regardless of num_features is one
    retry=degrade_step>0
    while retry:
        solution = milp(c, integrality=integrality, bounds=bool_bound, constraints=constraint)
        retry = solution.status == 2
        if retry:
            # reduce the lower bound by degrade step to attain feasability
            lower_divergence_bound = list(map(lambda x: x-degrade_step, lower_divergence_bound))
            constraint = LinearConstraint(A=pairwise_divergences, lb=lower_divergence_bound)
    feature_selection_vector = solution.x.astype(np.bool_)
    return feature_selection_vector

def grouped_distribution_divergence(dataset, groupings, event_sets, min_group_size=5, sampling_num=100, js_base=2.0, crossval_split=0, lower_divergence_bound=0.9):
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
    group_selections, retained_groups = select_grouping(dataset, groupings, min_group_size, crossval_split=crossval_split)
    if crossval_split <= 1:
        feature_wise_distributions = get_feature_wise_distribution(group_selections, list(groupings.keys()), event_sets, sampling_num)
        divergence_df = get_pairwise_feature_divergence(feature_wise_distributions, js_base=js_base)
        selected_feature_df, _ = select_features(divergence_df, lower_divergence_bound)
        return selected_feature_df
    else:
        results = {}
        for split_id in group_selections.keys():
            train_selection = group_selections[split_id]["train"]
            test_selection = group_selections[split_id]["test"]
            feature_wise_distributions = get_feature_wise_distribution(train_selection, list(groupings.keys()), event_sets, sampling_num)
            divergence_df = get_pairwise_feature_divergence(feature_wise_distributions, js_base=js_base)
            results[split_id] = {"div_df": divergence_df, "test": test_selection, "train": train_selection, "group_names": retained_groups}
        return results
    
def feature_analysis_pipe(raw_word_src="./data/features/word_histogram_union_raw.csv",
                          pruned_word_src="./data/features/word_histogram_union_pruned.csv",
                          sent_src="./data/features/sentence_lengths_raw.json",
                          metadata_src="./data/metadata.csv",
                          # TODO: find a good set of groups. maybe like 3 ish or so. author, country of origin and institution maybe?
                          groupby=["first_author","first_author_institution","first_author_country"],
                          normalize=True,
                          crossval_split=0,
                          lower_divergence_bound=0.9):
    union_raw_word_df = load_csv(raw_word_src)
    union_pruned_word_df = load_csv(pruned_word_src)
    # NOTE: max_len 50 is chosen here, because it is twice the commonly recommended max sentence lenght of 25
    raw_sentence_df, raw_sentence_dict = load_sentence_json(sent_src, max_len=45)
    metadata_df = load_metadata(metadata_src)

    # get supplementary features
    sentence_stats = get_mean_and_stdev_sent(raw_sentence_df)
    easy_word_ratios = get_easy_words_count(union_raw_word_df)[["easy_word_ratio"]]
    syllable_counts = get_syllable_counts(union_raw_word_df)
    # NOTE: choice of commonality theshhold was relatively arbitrary
    common_word_counts = get_common_word_df(union_pruned_word_df, commonality_thershhold=0.6, cutoff=-1)

    # normalize supplementary feature dfs where appropriate
    if normalize:
        syllable_ratios_array = normalize_word_rows(syllable_counts.to_numpy())
        syllable_ratios = pd.DataFrame(data=syllable_ratios_array, index=syllable_counts.index, columns=syllable_counts.columns)
        common_word_ratios_array = normalize_word_rows(common_word_counts.to_numpy())
        common_word_ratios = pd.DataFrame(data=common_word_ratios_array, index=common_word_counts.index, columns=common_word_counts.columns)
        data_set_handles = ["word_freq", "syllable_freq", "easy_freq", "sentence"]
        data_sets = [common_word_ratios, syllable_ratios, easy_word_ratios, sentence_stats]
    else:
        data_set_handles = ["word_count", "syllable_count", "easy_freq", "sentence"]
        data_sets = [common_word_counts, syllable_counts, easy_word_ratios, sentence_stats]

    # compute groupings
    groupings = get_groupings(metadata_df, groupby) # NOTE: This returns multiple groups collected in a dict of dicts. DO NOT PASS DIRECTLY TO grouped_distribution_divergence

    dataset = unify_data_sets(data_sets, data_set_handles)

    event_sets = calculate_bins(dataset, num_bins=8)

    result_dfs = {}

    for group_name in groupings.keys():
        distribution_divergence_df = grouped_distribution_divergence(dataset, 
                                                                    groupings=groupings[group_name],
                                                                    event_sets=event_sets,
                                                                    sampling_num=1000,
                                                                    min_group_size=1,
                                                                    crossval_split=crossval_split,
                                                                    lower_divergence_bound=lower_divergence_bound)
        result_dfs[group_name] = distribution_divergence_df

    return result_dfs

def select_and_predict(split_result):
    split_divergence_df = split_result["div_df"]
    split_holdout = split_result["test"]
    split_train = split_result["train"]
    split_group_names = split_result["group_names"]
    _, selected_features = select_features(split_divergence_df, lower_divergence_bound=0.9)
    train_dict = {}
    test_dict = {}
    for i, group_name_inner in enumerate(split_group_names):
        train_dict[group_name_inner] = split_train[i][selected_features]
        test_dict[group_name_inner] = split_holdout[i][selected_features].to_numpy()
    group_predictor = Group_Predictor()
    group_predictor.fit(train_dict)
    group_prediction = group_predictor.predict(test_dict)
    return group_prediction

def prediction_pipe(analysis_results=None,
                    raw_word_src="./data/features/word_histogram_union_raw.csv",
                    pruned_word_src="./data/features/word_histogram_union_pruned.csv",
                    sent_src="./data/features/sentence_lengths_raw.json",
                    metadata_src="./data/metadata.csv",
                    groupby=["first_author","first_author_institution","first_author_country"],
                    normalize=True,
                    crossval_split=10):
    if analysis_results is None:
        result_dict = feature_analysis_pipe(raw_word_src=raw_word_src,
                                        pruned_word_src=pruned_word_src,
                                        sent_src=sent_src,
                                        metadata_src=metadata_src,
                                        groupby=groupby,
                                        normalize=normalize,
                                        crossval_split=crossval_split)
    else:
        result_dict = analysis_results
    performances_dict = {}
    for group_name in result_dict.keys():
        split_result_dicts = result_dict[group_name]
        split_predictions = []

        for split in split_result_dicts.keys():
            split_result_dict = split_result_dicts[split]
            split_prediction = select_and_predict(split_result_dict)
            split_predictions.append(split_prediction)
        avg_performance = average_over_row_group(split_predictions)
        performances_dict[group_name] = avg_performance

    return performances_dict

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

class Group_Predictor:
    def __init__(self, bandwidth="silverman", kernel="gaussian"):
        self.bandwidth = bandwidth
        self.kernel = kernel

    def fit(self, sample_df_dict):
        self.group_names = list(sample_df_dict.keys())
        self.feature_names = list(sample_df_dict[self.group_names[0]].columns) # assume all dfs have the same cols
        sample_sets = [sample_df_dict[key].to_numpy() for key in self.group_names]
        all_samples = np.vstack(sample_sets)
        total_samples = all_samples.shape[0]
        self.p_groups = [sample_set.shape[0]/total_samples for sample_set in sample_sets]
        self.p_feats_given_group = [KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel).fit(sample_set) for sample_set in sample_sets]

    def predict(self, sample_array_dict):
        samples = np.vstack([sample_array_dict[key] for key in sample_array_dict.keys()])
        sample_ids = []
        for key in sample_array_dict.keys():
            sample_ids = sample_ids + [key]*sample_array_dict[key].shape[0]
        joined_group_probabilities = []
        for group_idx, group_name in enumerate(self.group_names):
            p_feat_group_joined = np.exp(self.p_feats_given_group[group_idx].score_samples(samples)) * self.p_groups[group_idx]
            joined_group_probabilities.append(np.expand_dims(p_feat_group_joined, axis=0))

        joined_group_probabilities_array = np.vstack(joined_group_probabilities)
        evidence = np.sum(joined_group_probabilities_array, axis=0)
        author_probs = joined_group_probabilities_array / evidence
        author_prob_df = pd.DataFrame(data=author_probs.T, index=sample_ids, columns=self.group_names)
        return author_prob_df

def main():
    feature_analysis_results = feature_analysis_pipe(groupby=["first_author", "first_author_country"])
    prediction_results = prediction_pipe(analysis_results=feature_analysis_results, groupby=["first_author", "first_author_country"])
    for key in prediction_results.keys():
        feature_analysis_result = feature_analysis_results[key]
        prediction_result = prediction_results[key]
        visualize_df_heatmap(feature_analysis_result, key + "_feature_analysis", None, (30,10), False)
        visualize_df_heatmap(prediction_result, key + "_prediction_performance", None, (30,10), True)
        

if __name__ == "__main__":
    main()