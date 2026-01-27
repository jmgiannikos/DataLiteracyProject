import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Set
import logging
import seaborn as sns
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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

def visualize_df_heatmap(data_df, title, save_path, figsize=(1,1), annot=False):
    fig, ax = plt.subplots(figsize=figsize)  # Set the figure size
    ax = sns.heatmap(data_df, ax=ax, annot=annot)
    ax.set_title(title)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved distribution plot to {save_path}")
    else:
        plt.show()