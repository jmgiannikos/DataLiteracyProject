import pandas as pd
import logging
from src.scrape_metadata import enrich_metadata_dataframe

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_enrichment():
    # Create sample data
    # 1. Paper with DOI (Transformer paper)
    # 2. Paper without DOI but known journal_ref (maybe?) - or just a recent one
    data = [
        {
            'arxiv_id': '1706.03762', 
            'title': 'Attention Is All You Need',
            'doi': '10.5555/3295222.3295349', # ACM DL link often used as DOI, or check arxiv metadata
            # Real DOI for Attention is All You Need is often listed as 10.5555/3295222.3295349 (Proceedings) or similar.
            # Let's use a paper with a clear Crossref DOI: ResNet '1512.03385' -> DOI 10.1109/CVPR.2016.90
            'journal_ref': None
        },
        {
            'arxiv_id': '1512.03385',
            'title': 'Deep Residual Learning for Image Recognition',
            'doi': '10.1109/CVPR.2016.90',
            'journal_ref': None
        }
    ]
    
    input_df = pd.DataFrame(data)
    
    # Run enrichment
    print("Running enrichment on test data...")
    result_df = enrich_metadata_dataframe(input_df, cache_dir="data/test_cache")
    
    print("\nResult columns:", result_df.columns.tolist())
    print("\nFirst row enriched data:")
    print(result_df.iloc[1][['arxiv_id', 'publication_year', 'cited_by_count', 'venue']])
    
    # Verification
    if 'cited_by_count' in result_df.columns:
        print("\nSUCCESS: Enrichment columns added.")
    else:
        print("\nFAILURE: Enrichment columns missing.")

if __name__ == "__main__":
    test_enrichment()
