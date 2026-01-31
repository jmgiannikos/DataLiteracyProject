# Pipeline initiator file
# (start) select authors
# -> select papers
# -> scrape metadata
# -> scrape text
# -> clean text
# -> extract features
# -> analysis
# visualization will be done in a separate jupyter notebook

from src.scrape_paper_ids import search_papers
from src.scrape_metadata import enrich_metadata_dataframe
from src.scrape_text import download_source, extract_text_from_source, CACHE_DIR, ensure_cache_dir, ARXIV_DELAY_LIMIT
from src.extract_features import batch_extract_features
from src.analysis import analyze_word_histograms
from src.utils import remove_duplicate_papers, merge_duplicate_authors, select_top_n_authors

import tempfile
import time
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# STEP 1 & 2: Search Papers & Enrich Metadata
# =============================================================================
METADATA_FILE = Path("data/top8-authors-category.csv")
SELECT_TOPN = True
metadata_dataframe = None

if METADATA_FILE.exists():
    print(f"\n[SKIP] Found existing metadata at {METADATA_FILE}. Skipping search and enrichment.")
    metadata_dataframe = pd.read_csv(METADATA_FILE)

    metadata_dataframe = merge_duplicate_authors(metadata_dataframe)
    # metadata_dataframe = remove_duplicate_papers(metadata_dataframe)
    if SELECT_TOPN:
        metadata_dataframe = select_top_n_authors(metadata_dataframe)

    metadata_dataframe.to_csv(METADATA_FILE, index=False)
    
    print(f"Loaded {len(metadata_dataframe)} papers from metadata.")
else:
    papers_df = search_papers(
            field="cs.LG",
            first_authorships=20,
            max_papers=1000,
            start_year=2010,
            end_year=2020
            )
        
    print(f"Testing with {len(papers_df)} arXiv papers:")
    print(papers_df.head())

    print("\nUnique first authors:")
    print(papers_df['first_author'].unique())
        
    papers_df.to_csv("data/test_papers.csv", index=False)

    metadata_dataframe = enrich_metadata_dataframe(papers_df)

    print(f"Processed {len(metadata_dataframe)} papers")
    print(f"\nDOIs found: {metadata_dataframe['doi'].notna().sum()}/{len(metadata_dataframe)}")
    print(f"\nMetadata sources:")
    print(metadata_dataframe['metadata_source'].value_counts())
    print(f"\nAverage completeness: {metadata_dataframe['metadata_completeness'].mean():.2%}")
    print(f"\nOutput saved to: data/metadata.csv")

# =============================================================================
# STEP 3 & 4: Scrape Text & Extract Features
# =============================================================================
output_dir = "data/features"
FEATURES_FILE = Path(output_dir) / "word_histogram_union_pruned.csv"

if FEATURES_FILE.exists():
    print(f"\n[SKIP] Found existing features at {FEATURES_FILE}. Skipping text scraping and feature extraction.")
else:
    # STEP 3: Scrape text from arXiv sources
    print("\n" + "="*60)
    print("STEP 3: Scraping text from arXiv sources")
    print("="*60)

    ensure_cache_dir()
    successful_extractions = 0

    arxiv_ids = metadata_dataframe['arxiv_id'].tolist()

    for i, arxiv_id in enumerate(arxiv_ids):
        # Sanitize arxiv_id for filename (replace slashes with underscores for old-style IDs like "astro-ph/9906233")
        safe_arxiv_id = str(arxiv_id).replace("/", "_")
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

    # STEP 4: Extract features from text
    print("\n" + "="*60)
    print("STEP 4: Extracting features from text")
    print("="*60)

    text_dir = str(CACHE_DIR)
    
    features = batch_extract_features(text_dir, output_dir, generate_csv=True, metadata_df=metadata_dataframe)

    print(f"\nFeature extraction complete: {len(features)} documents processed")
    print(f"Output saved to: {output_dir}/")

# =============================================================================
# STEP 5: Run analysis
# =============================================================================
# NOTE: currently outdated. Does not run a significat section of analysis
#print("\n" + "="*60)
#print("STEP 5: Running analysis")
#print("="*60)

#csv_path = f"{output_dir}/word_histogram_union_pruned.csv"
#analysis_output_dir = "data/analysis"

# Check if CSV exists before running analysis
#if Path(csv_path).exists():
#    results = analyze_word_histograms(csv_path, analysis_output_dir)
#    print(f"\nAnalysis complete:")
#    print(f"  - Documents analyzed: {results['n_documents']}")
#    print(f"  - Features (words): {results['n_features']}")
#    print(f"  - Unique authors: {len(results['unique_authors'])}")
#    print(f"  - PCA explained variance: {results['pca_explained_variance']}")
#    print(f"\nPlots saved to: {analysis_output_dir}/")
#else:
#    print(f"Warning: {csv_path} not found. Skipping analysis.")
#    print("This may occur if no valid text was extracted from any papers.")

print("\n" + "="*60)
print("PIPELINE COMPLETE")
print("="*60)
