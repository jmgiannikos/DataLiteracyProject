# Pipeline initiator file
# (start) select authors
# -> select papers
# -> scrape metadata
# -> scrape text
# -> clean text
# -> extract features
# -> analysis
# visualization will be done in a separate jupyter notebook

from src.scrape_paper_ids import get_papers
from src.scrape_metadata import scrape_metadata_arxivIDs
from src.scrape_text import download_source, extract_text_from_source, CACHE_DIR, ensure_cache_dir, ARXIV_DELAY_LIMIT
from src.extract_features import batch_extract_features
from src.analysis import analyze_word_histograms

import tempfile
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Physics Author
arxiv_ids_florentin = get_papers(
    entry="Florentin Millour",
    n=100,  # Get co-authors
    j=10000,  # first-author papers per author
    k=100   # non-first-author papers per author
)

# Biology Author
arxiv_ids_manel = get_papers(entry="Manel Gil-Sorribes",
    n=100,  # Get co-authors
    j=10000,  # first-author papers per author
    k=100   # non-first-author papers per author
)

# Econ Author
arxiv_ids_wen = get_papers(entry="Wen Lou",
    n=100,  # Get co-authors
    j=10000,  # first-author papers per author
    k=100   # non-first-author papers per author
)

arxiv_ids = arxiv_ids_florentin.union(arxiv_ids_manel).union(arxiv_ids_wen)

print(f"Testing with {len(arxiv_ids)} arXiv papers:")
for arxiv_id in arxiv_ids:
    print(f"  - {arxiv_id}")
print()

metadata_dataframe = scrape_metadata_arxivIDs(arxiv_ids)

print(f"Processed {len(metadata_dataframe)} papers")
print(f"\nDOIs found: {metadata_dataframe['doi'].notna().sum()}/{len(metadata_dataframe)}")
print(f"\nMetadata sources:")
print(metadata_dataframe['metadata_source'].value_counts())
print(f"\nAverage completeness: {metadata_dataframe['metadata_completeness'].mean():.2%}")
print(f"\nOutput saved to: data/metadata.csv")

# =============================================================================
# STEP 3: Scrape text from arXiv sources
# =============================================================================
print("\n" + "="*60)
print("STEP 3: Scraping text from arXiv sources")
print("="*60)

ensure_cache_dir()
successful_extractions = 0

for i, arxiv_id in enumerate(arxiv_ids):
    # Sanitize arxiv_id for filename (replace slashes with underscores for old-style IDs like "astro-ph/9906233")
    safe_arxiv_id = arxiv_id.replace("/", "_")
    target_file = CACHE_DIR / f"{safe_arxiv_id}.txt"

    if target_file.exists():
        logger.info(f"[{i+1}/{len(arxiv_ids)}] Skipping {arxiv_id}, already cached.")
        successful_extractions += 1
        continue

    logger.info(f"[{i+1}/{len(arxiv_ids)}] Processing {arxiv_id}...")

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=True) as temp_source:
        if download_source(arxiv_id, temp_source.name):
            # Check if it's a PDF
            with open(temp_source.name, 'rb') as f:
                header = f.read(4)

            if header == b'%PDF':
                logger.warning(f"Source for {arxiv_id} is PDF only. Skipping.")
                with open(target_file, 'w', encoding='utf-8') as f:
                    f.write("PDF_ONLY")
            else:
                text = extract_text_from_source(temp_source.name)
                if text:
                    with open(target_file, 'w', encoding='utf-8') as f:
                        f.write(text)
                    logger.info(f"Successfully extracted text for {arxiv_id}")
                    successful_extractions += 1
                else:
                    logger.warning(f"No text extracted for {arxiv_id}")

    # Rate limiting
    time.sleep(ARXIV_DELAY_LIMIT)

print(f"\nText extraction complete: {successful_extractions}/{len(arxiv_ids)} successful")

# =============================================================================
# STEP 4: Extract features from text
# =============================================================================
print("\n" + "="*60)
print("STEP 4: Extracting features from text")
print("="*60)

text_dir = str(CACHE_DIR)
output_dir = "data/features"

features = batch_extract_features(text_dir, output_dir, generate_csv=True)

print(f"\nFeature extraction complete: {len(features)} documents processed")
print(f"Output saved to: {output_dir}/")

# NOTE: currently outdated. Does not run a significat section of analysis
# =============================================================================
# STEP 5: Run analysis
# =============================================================================
print("\n" + "="*60)
print("STEP 5: Running analysis")
print("="*60)

csv_path = f"{output_dir}/word_histogram_union_pruned.csv"
analysis_output_dir = "data/analysis"

# Check if CSV exists before running analysis
if Path(csv_path).exists():
    results = analyze_word_histograms(csv_path, analysis_output_dir)
    print(f"\nAnalysis complete:")
    print(f"  - Documents analyzed: {results['n_documents']}")
    print(f"  - Features (words): {results['n_features']}")
    print(f"  - Unique authors: {len(results['unique_authors'])}")
    print(f"  - PCA explained variance: {results['pca_explained_variance']}")
    print(f"\nPlots saved to: {analysis_output_dir}/")
else:
    print(f"Warning: {csv_path} not found. Skipping analysis.")
    print("This may occur if no valid text was extracted from any papers.")

print("\n" + "="*60)
print("PIPELINE COMPLETE")
print("="*60)
