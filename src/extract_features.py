import statistics
import json
import csv
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
from sklearn.linear_model import TheilSenRegressor, RANSACRegressor
import pandas as pd
from os.path import isfile

# Optional dependency - graceful fallback if not installed
try:
    import enchant
    HAS_ENCHANT = True
except ImportError:
    HAS_ENCHANT = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


# =============================================================================
# SENTENCE ANALYSIS FUNCTIONS (from jan-analysis/csv_dataset_generator.py)
# =============================================================================

def get_sentence_len(sentence: str) -> int:
    """Get word count for a sentence."""
    words = word_tokenize(sentence, language='english')
    return len(words)

# NOTE: Works well enough
def remove_outlier_sentences(
    sentences: List[str],
    sentence_lengths: List[int],
    n: int = 4
) -> Tuple[List[str], List[int]]:
    """
    Remove sentences that are statistical outliers by length.

    Args:
        sentences: List of sentence strings
        sentence_lengths: Corresponding word counts
        n: Number of standard deviations for outlier threshold

    Returns:
        Tuple of (pruned_sentences, pruned_lengths)

    Source: jan-analysis/csv_dataset_generator.py
    """
    if len(sentence_lengths) <= 2:
        return sentences, sentence_lengths

    pruned_sentences = []
    pruned_lengths = []

    mean_len = statistics.mean(sentence_lengths)
    stdev_len = statistics.stdev(sentence_lengths)

    lower_bound = max([mean_len - stdev_len * n, 0])
    upper_bound = mean_len + stdev_len * n

    for sentence, length in zip(sentences, sentence_lengths):
        if lower_bound <= length <= upper_bound:
            pruned_sentences.append(sentence)
            pruned_lengths.append(length)

    logger.debug(f"Removed {len(sentences) - len(pruned_sentences)} outlier sentences")
    return pruned_sentences, pruned_lengths


# =============================================================================
# WORD HISTOGRAM FUNCTIONS (from jan-analysis/csv_dataset_generator.py)
# =============================================================================

def get_word_histogram(
    text_words: List[str],
    stem: bool = False,
    check_spelling: bool = True
) -> Dict[str, int]:
    """
    Build word frequency histogram with optional stemming and spell-checking.

    Args:
        text_words: List of word tokens
        stem: Apply Porter stemming if True
        check_spelling: Filter non-English words if True (requires PyEnchant)

    Returns:
        Dictionary mapping words to frequency counts

    Source: jan-analysis/csv_dataset_generator.py
    """
    word_hist = {}

    if stem:
        porter_stemmer = PorterStemmer()

    # Initialize spell checker if available and requested
    spell_dict = None
    if check_spelling and HAS_ENCHANT:
        try:
            spell_dict = enchant.Dict("en_US")
        except Exception as e:
            logger.warning(f"Could not initialize spell-checker: {e}. Continuing without.")
            spell_dict = None
    elif check_spelling and not HAS_ENCHANT:
        logger.debug("PyEnchant not installed. Spell-checking disabled.")

    for word in text_words:
        word_lower = word.lower()

        # Spell checking filter
        if spell_dict is not None:
            if not spell_dict.check(word_lower):
                continue
            # Skip single characters except 'a' and 'i'
            if len(word_lower) == 1 and word_lower not in ('a', 'i'):
                continue

        # Apply stemming
        if stem:
            word_lower = porter_stemmer.stem(word_lower)

        # Count
        if word_lower in word_hist:
            word_hist[word_lower] += 1
        else:
            word_hist[word_lower] = 1

    return word_hist


def check_zipfs_law(
    word_histogram: Dict[str, int],
    n: int = 10,
    verbose: bool = False
) -> Dict[str, int]:
    """
    Remove words that deviate significantly from Zipf's law distribution.

    Uses RANSAC regression (outlier-resistant) to fit expected frequencies,
    then removes words whose frequency deviates by more than n standard deviations.

    Args:
        word_histogram: Word to frequency mapping
        n: Number of standard deviations for outlier threshold
        verbose: Print removed words if True

    Returns:
        Filtered word histogram

    Source: jan-analysis/csv_dataset_generator.py
    """
    if len(word_histogram) <= 1:
        return word_histogram

    # Sort by frequency descending
    sortable_hist = [(count, word) for word, count in word_histogram.items()]
    sortable_hist.sort(reverse=True, key=lambda x: x[0])

    # According to Zipf's law, inverted frequency should be roughly linear with rank
    inverted_counts = [x[0] ** -1 for x in sortable_hist]

    # Fit outlier-resistant linear model
    estimator = RANSACRegressor(random_state=42)
    index_array = np.expand_dims(np.arange(len(inverted_counts), step=1), axis=1)
    print(inverted_counts)
    estimator.fit(index_array, inverted_counts)
    predicted_counts = estimator.predict(index_array)

    # Transform back to expected counts
    expected_counts = []
    for pred in predicted_counts:
        if pred > 0:
            expected_counts.append(max(int(round(pred ** -1)), 0))
        elif pred == 0:
            expected_counts.append(0)
        else:
            print(pred)
            expected_counts.append(0)

    # Compute normalized errors
    errors = []
    for i, expected in enumerate(expected_counts):
        actual = sortable_hist[i][0]
        errors.append((expected - actual) / actual)

    # Statistical thresholds (median is more outlier-resistant)
    median_error = statistics.median(errors)
    stdev_error = statistics.stdev(errors) if len(errors) > 1 else 0

    # Filter words
    remaining_words = {}
    removed_count = 0

    for i, error in enumerate(errors):
        word = sortable_hist[i][1]
        count = sortable_hist[i][0]

        lower_bound = median_error - n * stdev_error
        upper_bound = median_error + n * stdev_error

        if lower_bound <= error <= upper_bound:
            remaining_words[word] = count
        else:
            removed_count += 1
            #print(f"Removed word: {word} (count={count}, error={error:.2f})")
            if verbose:
                logger.info(f"Removed word: {word} (count={count}, error={error:.2f})")

    logger.debug(f"Zipf's law filter removed {removed_count} words")
    return remaining_words


def join_word_hists(
    word_hists: List[Dict[str, int]],
    union: bool = True
) -> List[Dict[str, int]]:
    """
    Unify word histograms to have the same vocabulary.

    Args:
        word_hists: List of word histogram dictionaries
        union: If True, use union of all words (fill missing with 0).
               If False, use intersection (only words in all documents).

    Returns:
        List of histograms with aligned vocabularies

    Source: jan-analysis/csv_dataset_generator.py
    """
    if not word_hists:
        return []

    # Build lexicon
    lexicon = []
    if union:
        for word_hist in word_hists:
            contained_words = list(word_hist.keys())
            lexicon = list(set(lexicon + contained_words))
    else:
        for i, word_hist in enumerate(word_hists):
            contained_words = list(word_hist.keys())
            if i == 0:
                lexicon = contained_words
            else:
                lexicon = list(set(lexicon) & set(contained_words))

    # Align histograms to lexicon
    joined_word_hists = []
    for word_hist in word_hists:
        joined_hist = {}
        for word in lexicon:
            joined_hist[word] = word_hist.get(word, 0)
        joined_word_hists.append(joined_hist)

    return joined_word_hists


def append_sentences(sentences: List[str]) -> str:
    """Concatenate sentences into single text string."""
    return " ".join(sentences)


# =============================================================================
# CSV/JSON GENERATION FUNCTIONS (from jan-analysis/csv_dataset_generator.py)
# =============================================================================

def transform_to_csv(
    data_handles: List[str],
    histograms: List[Dict[str, int]]
) -> List[List]:
    """Convert histograms to CSV row format."""
    if not histograms:
        return []

    col_order = list(histograms[0].keys())
    rows = [["Data Name"] + col_order]  # Header

    for i, hist in enumerate(histograms):
        row = [data_handles[i]] + [hist[word] for word in col_order]
        rows.append(row)

    return rows


def generate_sentence_json(
    sentence_lists: List[List[str]],
    data_handles: List[str]
) -> Tuple[Dict, Dict]:
    """
    Generate sentence length JSON for multiple documents.

    Args:
        sentence_lists: List of sentence lists (one per document)
        data_handles: Identifiers for each document

    Returns:
        Tuple of (pruned_json_dict, raw_json_dict)

    Source: jan-analysis/csv_dataset_generator.py
    """
    pruned_json_dict = {}
    raw_json_dict = {}

    for i, sentences in enumerate(sentence_lists):
        sentence_lengths = [get_sentence_len(s) for s in sentences]
        raw_json_dict[data_handles[i]] = sentence_lengths

        _, pruned_lengths = remove_outlier_sentences(sentences, sentence_lengths)
        pruned_json_dict[data_handles[i]] = pruned_lengths

    return pruned_json_dict, raw_json_dict


def generate_wordhist_csv(
    sentence_lists: List[List[str]],
    data_handles: List[str]
) -> Tuple[List, List, List, List]:
    """
    Generate word histogram CSV data for multiple documents.

    Args:
        sentence_lists: List of sentence lists (one per document)
        data_handles: Identifiers for each document

    Returns:
        Tuple of (union_pruned, inter_pruned, union_raw, inter_raw) CSV rows

    Source: jan-analysis/csv_dataset_generator.py
    """
    pruned_word_hists = []
    word_hists = []

    for sentences in sentence_lists:
        sentence_lengths = [get_sentence_len(s) for s in sentences]

        # Pruned version (outliers removed, stemmed, Zipf-filtered)
        pruned_sentences, _ = remove_outlier_sentences(sentences, sentence_lengths)
        pruned_text = append_sentences(pruned_sentences)
        pruned_words = word_tokenize(pruned_text, language='english')
        # NOTE: Zipfs law check ineffective. Removed.
        pruned_word_hist = get_word_histogram(pruned_words, stem=True)
        pruned_word_hists.append(pruned_word_hist)

        # Raw version (no outlier removal, no stemming)
        raw_text = append_sentences(sentences)
        words = word_tokenize(raw_text, language='english')
        word_hist = get_word_histogram(words, stem=False)
        word_hists.append(word_hist)

    # Create CSV variants
    csv_union_pruned = transform_to_csv(data_handles, join_word_hists(pruned_word_hists, union=True))
    csv_inter_pruned = transform_to_csv(data_handles, join_word_hists(pruned_word_hists, union=False))
    csv_union_raw = transform_to_csv(data_handles, join_word_hists(word_hists, union=True))
    csv_inter_raw = transform_to_csv(data_handles, join_word_hists(word_hists, union=False))

    return csv_union_pruned, csv_inter_pruned, csv_union_raw, csv_inter_raw


# =============================================================================
# HIGH-LEVEL EXTRACTION FUNCTIONS
# =============================================================================

def extract_features_from_text(
    text: str,
    document_id: str,
    stem: bool = True,
    check_spelling: bool = True,
    # NOTE: zipfs filter is not working currently. DO NOT ENABLE
    apply_zipf_filter: bool = False,
    remove_outliers: bool = True
) -> Dict:
    """
    Extract all features from a processed text document.

    Args:
        text: Processed plain text (from scrape_text.py output)
        document_id: Identifier for the document
        stem: Apply Porter stemming to words
        check_spelling: Filter non-English words
        apply_zipf_filter: Remove Zipf's law outliers
        remove_outliers: Remove outlier sentences by length

    Returns:
        Dictionary with all extracted features
    """
    # Tokenize into sentences
    sentences = sent_tokenize(text, language='english')
    sentence_lengths = [get_sentence_len(s) for s in sentences]

    # Optionally remove outlier sentences
    if remove_outliers and len(sentence_lengths) > 2:
        pruned_sentences, pruned_lengths = remove_outlier_sentences(sentences, sentence_lengths)
    else:
        pruned_sentences, pruned_lengths = sentences, sentence_lengths

    # Build word histogram
    pruned_text = append_sentences(pruned_sentences)
    words = word_tokenize(pruned_text, language='english')
    word_hist = get_word_histogram(words, stem=stem, check_spelling=check_spelling)

    # Optionally apply Zipf's law filter
    if apply_zipf_filter and len(word_hist) > 1:
        word_hist = check_zipfs_law(word_hist)

    # Compute statistics
    features = {
        'document_id': document_id,
        'total_sentences': len(sentences),
        'pruned_sentences': len(pruned_sentences),
        'total_words': sum(sentence_lengths),
        'pruned_words': sum(pruned_lengths),
        'unique_words': len(word_hist),
        'mean_sentence_length': statistics.mean(pruned_lengths) if pruned_lengths else 0,
        'stdev_sentence_length': statistics.stdev(pruned_lengths) if len(pruned_lengths) > 1 else 0,
        'word_histogram': word_hist,
        'sentence_lengths': pruned_lengths
    }

    return features


def batch_extract_features(
    text_dir: str,
    output_dir: str = "data/features",
    generate_csv: bool = True,
    metadata_df: pd.Dataframe = None
) -> List[Dict]:
    """
    Extract features from all text files in a directory.

    Args:
        text_dir: Directory containing .txt files
        output_dir: Directory to save feature JSON/CSV files
        generate_csv: Whether to generate combined CSV files

    Returns:
        List of feature dictionaries
    """
    text_path = Path(text_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if metadata_df is None:
        txt_files = list(text_path.glob("*.txt"))
    else:
        file_ids = metadata_df["arxiv_id"]
        txt_files = [Path(str(text_path) + "/" + str(file_id) + ".txt") for file_id in file_ids if isfile(str(text_path) + "/" + str(file_id) + ".txt")]
        assert len(txt_files) != 0
    logger.info(f"Processing {len(txt_files)} text files...")

    all_features = []
    sentence_lists = []
    data_handles = []

    for txt_file in txt_files:
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read()

            if text == "PDF_ONLY" or not text.strip():
                logger.warning(f"Skipping {txt_file.name}: no text content")
                continue

            doc_id = txt_file.stem
            features = extract_features_from_text(text, doc_id)
            all_features.append(features)

            # Collect for batch CSV generation
            sentences = sent_tokenize(text, language='english')
            sentence_lists.append(sentences)
            data_handles.append(doc_id)

            # Save individual feature file
            feature_file = output_path / f"{doc_id}_features.json"
            # Convert to JSON-serializable format (exclude large histogram)
            feature_summary = {k: v for k, v in features.items() if k != 'word_histogram'}
            feature_summary['vocabulary_size'] = len(features['word_histogram'])
            with open(feature_file, 'w', encoding='utf-8') as f:
                json.dump(feature_summary, f, indent=2)

            logger.info(f"Extracted features for {doc_id}")

        except Exception as e:
            logger.error(f"Failed to extract features from {txt_file}: {e}")

    # Generate combined outputs
    if all_features and generate_csv:
        logger.info("Generating combined CSV and JSON outputs...")

        # Word histogram CSVs
        try:
            csv_union_pruned, csv_inter_pruned, csv_union_raw, csv_inter_raw = generate_wordhist_csv(
                sentence_lists, data_handles
            )

            for name, rows in [
                ('union_pruned', csv_union_pruned),
                ('inter_pruned', csv_inter_pruned),
                ('union_raw', csv_union_raw),
                ('inter_raw', csv_inter_raw)
            ]:
                csv_file = output_path / f"word_histogram_{name}.csv"
                with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerows(rows)
                logger.info(f"Saved {csv_file}")
        except Exception as e:
            logger.error(f"Failed to generate word histogram CSVs: {e}")

        # Sentence length JSONs
        try:
            pruned_json, raw_json = generate_sentence_json(sentence_lists, data_handles)
            for name, data in [('pruned', pruned_json), ('raw', raw_json)]:
                json_file = output_path / f"sentence_lengths_{name}.json"
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                logger.info(f"Saved {json_file}")
        except Exception as e:
            logger.error(f"Failed to generate sentence length JSONs: {e}")

    logger.info(f"Extracted features from {len(all_features)} documents")
    return all_features

def main():
    import sys 

    if len(sys.argv) > 1:
        text_dir = sys.argv[1]
    else:
        text_dir = "src/data/cache/raw_text"

    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = "src/data/features"

    batch_extract_features(text_dir, output_dir)

if __name__ == "__main__":
    main()
