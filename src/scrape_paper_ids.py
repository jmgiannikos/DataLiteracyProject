import arxiv
import logging
import pandas as pd
import os
from collections import Counter
from typing import List, Dict, Any, Set, Optional, Union
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default max results to fetch from arxiv
MAX_RESULTS = 200

def normalize_name(name: str) -> str:
    """
    Removes periods, spaces, and commas, and converts to lowercase.
    """
    return name.lower().replace(".", "").replace(" ", "").replace(",", "")

def extract_arxiv_id(arxiv_input: str) -> str:
    """
    Extract clean arXiv ID from various input formats.
    - Clean IDs: "2103.00020", "1706.03762"
    - With version: "2103.00020v1", "1706.03762v2"
    - URLs: "http://arxiv.org/abs/2103.00020v1", "http://arxiv.org/abs/astro-ph/9906233v1"
    - Old style IDs: "astro-ph/9906233"

    Returns:
    - Clean ID without version: "2103.00020", "astro-ph/9906233"
    """
    # Regex to find arXiv IDs (new style or old style)
    # New style: 4 digits, dot, 4-5 digits
    # Old style: Category (letters, dot, hyphen), slash, 7 digits
    pattern = r'((?:\d{4}\.\d{4,5})|(?:[a-zA-Z\-\.]+\/\d{7}))(?:v\d+)?'
    
    match = re.search(pattern, arxiv_input)
    if match:
        return match.group(1)

    # If no pattern matches, return as-is
    logger.warning(f"Could not parse arXiv ID from: {arxiv_input}")
    return arxiv_input

def get_papers_by_author(author_name: str, max_results: int = MAX_RESULTS) -> List[arxiv.Result]:
    """
    Fetches papers for a given author using the arXiv API.
    Arg:

    """
    search = arxiv.Search(
        query=f'au:"{author_name}"',
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    client = arxiv.Client()
    results = []
    try:
        results = list(client.results(search))
    except Exception as e:
        logger.error(f"Error fetching papers for {author_name}: {e}")
    return results

def is_first_author(paper: arxiv.Result, author_name: str) -> bool:
    """
    Checks if the given author is the first author of the paper.
    Checks index 0 of the authors list.
    Approximate match for name comparison (bidirectional substring match).
    """
    if not paper.authors:
        return False

    normalized_query = normalize_name(author_name)
    normalized_first = normalize_name(paper.authors[0].name)

    # Bidirectional substring match to handle different name orderings
    # e.g., "Adya, V B" vs "V. B. Adya"
    return normalized_query in normalized_first or normalized_first in normalized_query

def get_papers_by_first_author(author_name: str, max_results: int = MAX_RESULTS) -> List[arxiv.Result]:
    """
    Get all first-author papers for a given author.

    Args:
        author_name: Author name to search for
        max_results: Maximum papers to fetch from arXiv

    Returns:
        List of arxiv.Result objects where author is first author

    Source: retrieve-data/arxiv_handler.py
    """
    papers = get_papers_by_author(author_name, max_results)
    return [p for p in papers if is_first_author(p, author_name)]

def is_in_date_range(paper: arxiv.Result, start_year: Optional[int], end_year: Optional[int]) -> bool:
    """Check if paper publication year is within range (inclusive)."""
    if start_year is not None and paper.published.year < start_year:
        return False
    if end_year is not None and paper.published.year > end_year:
        return False
    return True

def search_papers(
    field: Union[str, List[str]],
    first_authorships: int,
    max_papers: int,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    existing_csv_path: Optional[str] = None,
    incremental_save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Args:
        field: arXiv category/field (e.g. 'cs.AI') or list of categories for OR query
        first_authorships: Min number of first-author papers to qualify an author
        max_papers: Max total papers to collect
        start_year: Optional start year filter
        end_year: Optional end year filter
        existing_csv_path: Path to CSV containing already scraped papers to skip
        incremental_save_path: Path to CSV to save progress incrementally

    Returns:
        DataFrame with metadata for selected papers
    """
    # Handle field display for logging
    if isinstance(field, list):
        field_display = f"{len(field)} categories ({field[0]}...)"
    else:
        field_display = field
    logger.info(f"Searching for papers in field '{field_display}' (years: {start_year}-{end_year})")
    
    existing_df = pd.DataFrame()
    existing_ids = set()
    if existing_csv_path and os.path.exists(existing_csv_path):
        try:
            existing_df = pd.read_csv(existing_csv_path)
            if 'arxiv_id' in existing_df.columns:
                existing_ids = set(existing_df['arxiv_id'].astype(str))
                logger.info(f"Loaded {len(existing_ids)} existing arXiv IDs from {existing_csv_path}")
        except Exception as e:
            logger.warning(f"Could not load existing CSV: {e}")

    logger.info(f"Goal: {max_papers} NEW papers from authors with >{first_authorships} first-author papers")
    
    client = arxiv.Client()

    # Construct query with date filtering if needed
    if isinstance(field, list):
        cat_query = " OR ".join([f'cat:{f}' for f in field])
        query_parts = [f'({cat_query})']
    else:
        query_parts = [f'cat:{field}']
    
    if start_year is not None or end_year is not None:
        start_str = f"{start_year}01010000" if start_year else "000001010000"
        end_str = f"{end_year}12312359" if end_year else "209912312359"
        query_parts.append(f"submittedDate:[{start_str} TO {end_str}]")
        
    final_query = " AND ".join(query_parts)
    logger.info(f"ArXiv Query: {final_query}")

    search = arxiv.Search(
        query=final_query,
        max_results=max_papers * 50,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    checked_authors = set()
    collected_papers_data = []
    collected_ids = set()
    
    results_generator = client.results(search)
    
    def save_incremental():
        if not incremental_save_path:
            return
        logger.info(f"Incrementally saving {len(collected_papers_data)} papers to {incremental_save_path}")
        new_df = pd.DataFrame(collected_papers_data)
        if not existing_df.empty:
            combined = pd.concat([existing_df, new_df], ignore_index=True)
            combined.drop_duplicates(subset=['arxiv_id'], inplace=True)
        else:
            combined = new_df
        combined.to_csv(incremental_save_path, index=False)

    try:
        for paper in results_generator:
            if len(collected_papers_data) >= max_papers:
                break
                
            if not paper.authors:
                continue
                
            for author in paper.authors:
                if len(collected_papers_data) >= max_papers:
                    break
                    
                norm_name = normalize_name(author.name)
                if norm_name in checked_authors:
                    continue
                
                checked_authors.add(norm_name)
                
                try:
                    # Get potential papers for this author
                    author_papers = get_papers_by_first_author(author.name)
                    
                    # Filter these papers by date
                    valid_author_papers = [
                        p for p in author_papers 
                        if is_in_date_range(p, start_year, end_year)
                    ]
                    
                    # Check condition on filtered papers
                    if len(valid_author_papers) > first_authorships:
                        logger.info(f"Author {author.name} qualifies ({len(valid_author_papers)} papers)")
                        
                        # Add to results
                        newly_added_this_author = 0
                        for p in valid_author_papers:
                            if len(collected_papers_data) >= max_papers:
                                break
                            
                            p_id = extract_arxiv_id(p.entry_id)
                            
                            # Filter duplicates
                            if p_id in collected_ids or p_id in existing_ids:
                                continue
                                
                            collected_ids.add(p_id)
                            
                            # Collect metadata
                            paper_data = {
                                'arxiv_id': p_id,
                                'title': p.title,
                                'authors': [a.name for a in p.authors],
                                'first_author': p.authors[0].name if p.authors else None,
                                'summary': p.summary,
                                'published': p.published,
                                'doi': p.doi,
                                'primary_category': p.primary_category,
                                'categories': p.categories,
                                'pdf_url': p.pdf_url,
                                'journal_ref': p.journal_ref,
                                'updated': p.updated
                            }
                            collected_papers_data.append(paper_data)
                            newly_added_this_author += 1
                            
                            # Incremental save every 100 papers
                            if len(collected_papers_data) % 100 == 0:
                                save_incremental()

                except Exception as e:
                    logger.error(f"Error processing author {author.name}: {e}")
                    continue
                    
    except Exception as e:
        logger.error(f"Error during search iteration: {e}")

    # Final save
    save_incremental()
    logger.info(f"Returning metadata for {len(collected_papers_data)} unique papers")
    return pd.DataFrame(collected_papers_data)

if __name__ == "__main__":

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
    
    papers_df.to_csv("../data/test_papers.csv", index=False)
