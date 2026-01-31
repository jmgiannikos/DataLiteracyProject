# Pipeline initiator file for multi-category arXiv data collection
# Pipeline steps:
# 1. Collect papers from 5 arXiv category groups across 16 years
# 2. Merge all intermediate files into raw dataset
# 3. Enrich metadata with institutional/citation information
# 4. Clean data (remove incomplete rows and authors with few papers)
# 5. Extract text and compute features
# 6. Run analysis (optional)

from src.scrape_paper_ids import search_papers
from src.scrape_metadata import enrich_metadata_dataframe
from src.scrape_text import download_source, extract_text_from_source, CACHE_DIR, ensure_cache_dir, ARXIV_DELAY_LIMIT
from src.extract_features import batch_extract_features
from src.analysis import analyze_word_histograms

import tempfile
import os
import time
import logging
import argparse
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Explicit subcategory lists (wildcards may not work in arXiv API)
CATEGORY_GROUPS = {
    "cs": ["cs.AI", "cs.AR", "cs.CC", "cs.CE", "cs.CG", "cs.CL", "cs.CR", "cs.CV",
           "cs.CY", "cs.DB", "cs.DC", "cs.DL", "cs.DM", "cs.DS", "cs.ET", "cs.FL",
           "cs.GL", "cs.GR", "cs.GT", "cs.HC", "cs.IR", "cs.IT", "cs.LG", "cs.LO",
           "cs.MA", "cs.MM", "cs.MS", "cs.NA", "cs.NE", "cs.NI", "cs.OH", "cs.OS",
           "cs.PF", "cs.PL", "cs.RO", "cs.SC", "cs.SD", "cs.SE", "cs.SI", "cs.SY"],
    "econ": ["econ.EM", "econ.GN", "econ.TH"],
    "eess": ["eess.AS", "eess.IV", "eess.SP", "eess.SY"],
    "math": ["math.AC", "math.AG", "math.AP", "math.AT", "math.CA", "math.CO",
             "math.CT", "math.CV", "math.DG", "math.DS", "math.FA", "math.GM",
             "math.GN", "math.GR", "math.GT", "math.HO", "math.IT", "math.KT",
             "math.LO", "math.MG", "math.MP", "math.NA", "math.NT", "math.OA",
             "math.OC", "math.PR", "math.QA", "math.RA", "math.RT", "math.SG",
             "math.SP", "math.ST"],
    "physics": ["physics.acc-ph", "physics.ao-ph", "physics.app-ph", "physics.atm-clus",
                "physics.atom-ph", "physics.bio-ph", "physics.chem-ph", "physics.class-ph",
                "physics.comp-ph", "physics.data-an", "physics.ed-ph", "physics.flu-dyn",
                "physics.gen-ph", "physics.geo-ph", "physics.hist-ph", "physics.ins-det",
                "physics.med-ph", "physics.optics", "physics.plasm-ph", "physics.pop-ph",
                "physics.soc-ph", "physics.space-ph"]
}

# Years to collect (2010 through 2025)
YEARS = list(range(2010, 2026))

# Filtering parameters
MIN_FIRST_AUTHORSHIPS = 20  # Author must have > this many first-author papers
PAPERS_PER_TOPIC = 3000     # Target papers per broad topic across all years
MIN_PAPERS_PER_AUTHOR = 10  # For cleaning step
MIN_COMPLETENESS = 0      # Minimum metadata completeness score

# Output file paths
DATA_DIR = Path("data")
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
RAW_DATASET_PATH = DATA_DIR / "raw-arxiv-dataset.csv"
ENRICHED_DATASET_PATH = DATA_DIR / "enriched-arxiv-dataset.csv"
CLEANED_DATASET_PATH = DATA_DIR / "cleaned-dataset.csv"
PROCESSED_DATASET_PATH = DATA_DIR / "processed-dataset.csv"
FEATURES_DIR = DATA_DIR / "features"
ANALYSIS_DIR = DATA_DIR / "analysis"


# =============================================================================
# STEP 1: Collect Papers
# =============================================================================
def step1_collect_papers():
    """
    Collect papers for each category group across the entire date range.
    Saves and resumes from data/intermediate/{category}/combined.csv
    """
    logger.info(f"STEP 1: Collecting {PAPERS_PER_TOPIC} papers per topic from arXiv (2010-2025)")

    for category_name, subcategories in CATEGORY_GROUPS.items():
        category_dir = INTERMEDIATE_DIR / category_name
        category_dir.mkdir(parents=True, exist_ok=True)

        combined_path = category_dir / "combined.csv"
        
        existing_df = pd.DataFrame()
        if combined_path.exists():
            try:
                existing_df = pd.read_csv(combined_path)
                if len(existing_df) >= PAPERS_PER_TOPIC:
                    logger.info(f"[SKIP] {category_name} already has {len(existing_df)} papers")
                    continue
                logger.info(f"[RESUME] {category_name} has {len(existing_df)} papers. Target: {PAPERS_PER_TOPIC}")
            except Exception as e:
                logger.warning(f"Could not read {combined_path}, starting fresh: {e}")

        needed = PAPERS_PER_TOPIC - len(existing_df)
        logger.info(f"Collecting {needed} more papers for {category_name}...")

        try:
            new_papers_df = search_papers(
                field=subcategories,
                first_authorships=MIN_FIRST_AUTHORSHIPS,
                max_papers=needed,
                start_year=YEARS[0],
                end_year=YEARS[-1],
                existing_csv_path=str(combined_path) if combined_path.exists() else None,
                incremental_save_path=str(combined_path)
            )

            if len(new_papers_df) > 0:
                combined_df = pd.concat([existing_df, new_papers_df], ignore_index=True)
                combined_df.drop_duplicates(subset=['arxiv_id'], inplace=True)
                combined_df.to_csv(combined_path, index=False)
                logger.info(f"Updated {combined_path}: now has {len(combined_df)} papers")
            else:
                if len(existing_df) == 0:
                    logger.warning(f"No papers found for {category_name}")
                else:
                    logger.info(f"No additional papers found for {category_name}")

        except Exception as e:
            logger.error(f"Error collecting {category_name}: {e}")
            continue


# =============================================================================
# STEP 2: Merge to Raw Dataset
# =============================================================================
def step2_merge_to_raw_dataset(force=False):
    """
    Merge all intermediate files into a single raw dataset.
    Phase 1: Merge years within each category
    Phase 2: Merge all categories together
    """
    logger.info("STEP 2: Merging to raw dataset")

    if RAW_DATASET_PATH.exists() and not force:
        logger.info(f"[SKIP] {RAW_DATASET_PATH} already exists. Use force=True to rebuild.")
        return pd.read_csv(RAW_DATASET_PATH)

    all_category_dfs = []

    for category_name in CATEGORY_GROUPS.keys():
        category_combined_path = INTERMEDIATE_DIR / category_name / "combined.csv"

        if category_combined_path.exists():
            category_df = pd.read_csv(category_combined_path)
            category_df['source_category'] = category_name
            all_category_dfs.append(category_df)
            logger.info(f"Loaded {len(category_df)} papers from {category_name}")
        else:
            logger.warning(f"No combined file found for {category_name}")

    if not all_category_dfs:
        logger.error("No category data found to merge")
        return pd.DataFrame()

    # Merge all categories
    raw_dataset = pd.concat(all_category_dfs, ignore_index=True)

    # Remove duplicates across categories
    before_dedup = len(raw_dataset)
    raw_dataset.drop_duplicates(subset=['arxiv_id'], keep='first', inplace=True)
    after_dedup = len(raw_dataset)
    logger.info(f"Removed {before_dedup - after_dedup} duplicate papers across categories")

    # Save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_dataset.to_csv(RAW_DATASET_PATH, index=False)
    logger.info(f"Saved raw dataset: {len(raw_dataset)} papers to {RAW_DATASET_PATH}")

    return raw_dataset


# =============================================================================
# STEP 3: Enrich Metadata
# =============================================================================
def step3_enrich_metadata():
    """
    Enrich the raw dataset with institutional/citation metadata.
    """
    logger.info("STEP 3: Enriching metadata")

    if ENRICHED_DATASET_PATH.exists():
        logger.info(f"[SKIP] {ENRICHED_DATASET_PATH} already exists")
        return pd.read_csv(ENRICHED_DATASET_PATH)

    if not RAW_DATASET_PATH.exists():
        logger.error(f"Raw dataset not found at {RAW_DATASET_PATH}")
        return None

    raw_df = pd.read_csv(RAW_DATASET_PATH)
    logger.info(f"Enriching metadata for {len(raw_df)} papers...")

    enriched_df = enrich_metadata_dataframe(
        input_df=raw_df,
        cache_dir=str(DATA_DIR / "cache"),
        output_path=str(ENRICHED_DATASET_PATH)
    )

    logger.info(f"Enrichment complete. Saved to {ENRICHED_DATASET_PATH}")
    logger.info(f"DOIs found: {enriched_df['doi'].notna().sum()}/{len(enriched_df)}")
    logger.info(f"Metadata sources: {enriched_df['metadata_source'].value_counts().to_dict()}")
    logger.info(f"Average completeness: {enriched_df['metadata_completeness'].mean():.2%}")

    return enriched_df


# =============================================================================
# STEP 4: Clean Data
# =============================================================================
def step4_clean_data():
    """
    Clean the enriched dataset:
    1. Remove rows with metadata_completeness < MIN_COMPLETENESS
    2. Remove authors with fewer than MIN_PAPERS_PER_AUTHOR papers
    """
    logger.info("STEP 4: Cleaning data")

    if CLEANED_DATASET_PATH.exists():
        logger.info(f"[SKIP] {CLEANED_DATASET_PATH} already exists")
        return pd.read_csv(CLEANED_DATASET_PATH)

    if not ENRICHED_DATASET_PATH.exists():
        logger.error(f"Enriched dataset not found at {ENRICHED_DATASET_PATH}")
        return None

    df = pd.read_csv(ENRICHED_DATASET_PATH)
    initial_count = len(df)
    logger.info(f"Starting cleaning with {initial_count} papers")

    # Step 1: Remove rows with low metadata completeness
    before_completeness = len(df)
    df = df[df['metadata_completeness'] >= MIN_COMPLETENESS]
    removed_completeness = before_completeness - len(df)
    logger.info(f"Removed {removed_completeness} rows with completeness < {MIN_COMPLETENESS}")

    # Step 2: Remove authors with fewer than MIN_PAPERS_PER_AUTHOR papers
    author_counts = df['first_author'].value_counts()
    valid_authors = author_counts[author_counts >= MIN_PAPERS_PER_AUTHOR].index

    before_author_filter = len(df)
    df = df[df['first_author'].isin(valid_authors)]
    after_author_filter = len(df)

    removed_authors = len(author_counts) - len(valid_authors)
    removed_papers = before_author_filter - after_author_filter
    logger.info(f"Removed {removed_authors} authors with <{MIN_PAPERS_PER_AUTHOR} papers")
    logger.info(f"Removed {removed_papers} papers from those authors")

    # Save cleaned dataset
    df.to_csv(CLEANED_DATASET_PATH, index=False)
    logger.info(f"Cleaned dataset: {len(df)} papers ({len(valid_authors)} authors) saved to {CLEANED_DATASET_PATH}")

    return df


# =============================================================================
# STEP 5: Extract Features
# =============================================================================
def step5_extract_features(cleaned_df, start_index=0):
    """
    Extract text and compute features from papers.
    5a: Scrape text from arXiv sources
    5b: Extract word histograms and other features
    5c: Create processed dataset
    """
    logger.info(f"STEP 5: Extracting features (starting from index {start_index})")

    if PROCESSED_DATASET_PATH.exists():
        logger.info(f"[SKIP] {PROCESSED_DATASET_PATH} already exists")
        return pd.read_csv(PROCESSED_DATASET_PATH)

    if cleaned_df is None or len(cleaned_df) == 0:
        logger.error("No cleaned data available for feature extraction")
        return None

    # Step 5a: Scrape text
    logger.info("Step 5a: Scraping text from arXiv sources")
    ensure_cache_dir()

    arxiv_ids = cleaned_df['arxiv_id'].tolist()
    
    # Manual override: slice the list if start_index > 0
    total_total = len(arxiv_ids)
    if start_index > 0:
        logger.info(f"Manual override: Skipping first {start_index} papers. Processing from index {start_index}...")
        arxiv_ids = arxiv_ids[start_index:]
        
    successful_extractions = 0

    for i, arxiv_id in enumerate(arxiv_ids):
        current_idx = i + start_index
        safe_arxiv_id = str(arxiv_id).replace("/", "_")
        target_file = CACHE_DIR / f"{safe_arxiv_id}.txt"

        if target_file.exists():
            # Check content for logs/debugging if needed, but skip regardless
            logger.info(f"[{current_idx+1}/{total_total}] Skipping {arxiv_id}, already attempted (cached)")
            successful_extractions += 1
            continue

        logger.info(f"[{current_idx+1}/{total_total}] Processing {arxiv_id}...")

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=True) as temp_source:
            if download_source(arxiv_id, temp_source.name):
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
                        logger.warning(f"No text extracted for {arxiv_id}. Marking as empty.")
                        with open(target_file, 'w', encoding='utf-8') as f:
                            f.write("EMPTY_TEXT")
            else:
                logger.error(f"Download failed for {arxiv_id}. Marking as failed.")
                with open(target_file, 'w', encoding='utf-8') as f:
                    f.write("FAILED_DOWNLOAD")

        time.sleep(ARXIV_DELAY_LIMIT)

    logger.info(f"Text extraction complete: {successful_extractions}/{len(arxiv_ids)} successful")

    # Step 5b: Extract features
    logger.info("Step 5b: Extracting features from text")

    features_csv = FEATURES_DIR / "word_histogram_union_pruned.csv"

    if not features_csv.exists():
        FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        features = batch_extract_features(str(CACHE_DIR), str(FEATURES_DIR), generate_csv=True)
        logger.info(f"Feature extraction complete: {len(features)} documents processed")
    else:
        logger.info(f"Features already exist at {features_csv}")

    # Step 5c: Create processed dataset
    logger.info("Step 5c: Creating processed dataset")

    if features_csv.exists():
        features_df = pd.read_csv(features_csv, index_col=0)

        # Create mapping from arxiv_id to features
        cleaned_df['safe_arxiv_id'] = cleaned_df['arxiv_id'].astype(str).str.replace("/", "_")

        # Keep only papers that have features
        valid_ids = set(features_df.index.astype(str))
        processed_df = cleaned_df[cleaned_df['safe_arxiv_id'].isin(valid_ids)].copy()
        processed_df['has_features'] = True

        processed_df.to_csv(PROCESSED_DATASET_PATH, index=False)
        logger.info(f"Processed dataset: {len(processed_df)} papers with features saved to {PROCESSED_DATASET_PATH}")

        return processed_df
    else:
        logger.error("Features CSV not found after extraction")
        return None


# =============================================================================
# STEP 6: Run Analysis (optional)
# =============================================================================
def step6_run_analysis():
    """
    Run analysis on the processed dataset.
    """
    logger.info("STEP 6: Running analysis")

    features_csv = FEATURES_DIR / "word_histogram_union_pruned.csv"

    if not features_csv.exists():
        logger.warning(f"Features CSV not found at {features_csv}. Skipping analysis.")
        return None

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    results = analyze_word_histograms(str(features_csv), str(ANALYSIS_DIR))

    logger.info(f"Analysis complete:")
    logger.info(f"  - Documents analyzed: {results['n_documents']}")
    logger.info(f"  - Features (words): {results['n_features']}")
    logger.info(f"  - Unique authors: {len(results['unique_authors'])}")
    logger.info(f"  - PCA explained variance: {results['pca_explained_variance']}")
    logger.info(f"Plots saved to: {ANALYSIS_DIR}/")

    return results


# =============================================================================
# STEP 7: Standalone Feature Extraction
# =============================================================================
def step7_standalone_feature_extraction():
    """
    Apply batch_extract_features to all currently available text in the cache.
    """
    logger.info("STEP 7: Standalone feature extraction from text cache")
    ensure_cache_dir()
    
    if not CACHE_DIR.exists() or not list(CACHE_DIR.glob("*.txt")):
        logger.warning(f"No text files found in {CACHE_DIR}. Skipping.")
        return None

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    features = batch_extract_features(str(CACHE_DIR), str(FEATURES_DIR), generate_csv=True)
    logger.info(f"Standalone feature extraction complete: {len(features)} documents processed")
    return features


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main():
    """
    Run the complete arXiv data pipeline.
    Each step checks if output exists and skips if so.
    """
    parser = argparse.ArgumentParser(description="arXiv Data Pipeline")
    parser.add_argument("-s", "--start-index", type=int, default=0, help="Index of paper to start from in Step 5 (default: 0)")
    parser.add_argument("--skip-to", type=int, choices=range(1, 8), help="Skip directly to a specific step (1-7)")
    parser.add_argument("--features-only", action="store_true", help="Shortcut to run only Step 7 (standalone feature extraction)")
    args = parser.parse_args()

    # Handle shortcut
    if args.features_only:
        args.skip_to = 7

    skip_to = args.skip_to if args.skip_to else 1

    print("=" * 60)
    print("ARXIV DATA PIPELINE - Starting")
    print("=" * 60)
    print(f"Categories: {list(CATEGORY_GROUPS.keys())}")
    print(f"Years: {YEARS[0]} to {YEARS[-1]}")
    if args.start_index > 0:
        print(f"Manual Start Index: {args.start_index}")
    if skip_to > 1:
        print(f"Skipping to Step {skip_to}")

    # Step 1: Collect papers
    if skip_to <= 1:
        print("\n" + "=" * 60)
        print("STEP 1: Collecting papers from arXiv")
        print("=" * 60)
        step1_collect_papers()

    # Step 2: Merge to raw dataset
    raw_df = None
    if skip_to <= 2:
        print("\n" + "=" * 60)
        print("STEP 2: Merging to raw dataset")
        print("=" * 60)
        raw_df = step2_merge_to_raw_dataset()

    # Step 3: Enrich metadata
    enriched_df = None
    if skip_to <= 3:
        print("\n" + "=" * 60)
        print("STEP 3: Enriching metadata")
        print("=" * 60)
        enriched_df = step3_enrich_metadata()

    # Step 4: Clean data
    cleaned_df = None
    if skip_to <= 4:
        print("\n" + "=" * 60)
        print("STEP 4: Cleaning data")
        print("=" * 60)
        cleaned_df = step4_clean_data()
    elif skip_to <= 5:
        # Need cleaned_df for step 5
        if CLEANED_DATASET_PATH.exists():
            cleaned_df = pd.read_csv(CLEANED_DATASET_PATH)
        else:
            logger.error(f"Cannot skip to step 5: {CLEANED_DATASET_PATH} not found.")
            return

    # Step 5: Extract features
    processed_df = None
    if skip_to <= 5:
        print("\n" + "=" * 60)
        print("STEP 5: Extracting features")
        print("=" * 60)
        processed_df = step5_extract_features(cleaned_df, start_index=args.start_index)

    # Step 6: Run analysis
    if skip_to <= 6:
        print("\n" + "=" * 60)
        print("STEP 6: Running analysis")
        print("=" * 60)
        step6_run_analysis()

    # Step 7: Standalone Feature Extraction
    if skip_to <= 7:
        print("\n" + "=" * 60)
        print("STEP 7: Standalone Feature Extraction (Batch)")
        print("=" * 60)
        step7_standalone_feature_extraction()

    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print("\nOutput files:")
    for path in [RAW_DATASET_PATH, ENRICHED_DATASET_PATH, CLEANED_DATASET_PATH, PROCESSED_DATASET_PATH]:
        if path.exists():
            df = pd.read_csv(path)
            print(f"  {path}: {len(df)} rows")
        else:
            print(f"  {path}: NOT CREATED")


if __name__ == "__main__":
    main()
