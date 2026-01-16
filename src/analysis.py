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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

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


# =============================================================================
# HIGH-LEVEL ANALYSIS FUNCTIONS
# =============================================================================

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
        csv_path = "data/features/word_histogram_union_pruned.csv"

    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    results = analyze_word_histograms(csv_path, output_dir)
    print(f"Analysis complete: {results}")
