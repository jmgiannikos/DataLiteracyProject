import arxiv
import logging
from collections import Counter
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default max results to fetch from arxiv
MAX_RESULTS = 200

def get_papers_by_author(author_name: str, max_results: int = MAX_RESULTS) -> List[arxiv.Result]:
    """
    Fetches papers for a given author using the arXiv API.
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
    first_author = paper.authors[0].name

    def normalize(name):
        return name.lower().replace(".", "").replace(" ", "").replace(",", "")

    normalized_query = normalize(author_name)
    normalized_first = normalize(first_author)

    # Bidirectional substring match to handle different name orderings
    # e.g., "Adya, V B" vs "V. B. Adya"
    return normalized_query in normalized_first or normalized_first in normalized_query


def get_co_authors(entry: str, n: int) -> List[str]:
    """
    Get the top n co-authors for the entry author.
    Args:
        entry: Entry point author name
        n: Number of co-authors to select

    Returns:
        List of n co-author names
    """
    logger.info(f"Finding top {n} co-authors for {entry}")

    papers = get_papers_by_author(entry)
    logger.info(f"Fetched {len(papers)} papers for {entry}")

    # Count co-authors
    co_author_counts = Counter()
    for paper in papers:
        for author in paper.authors:
            name = author.name
            # Skip entry author
            if name == entry or entry.lower() in name.lower():
                continue
            co_author_counts[name] += 1

    # Return top n
    top_co_authors = [name for name, _ in co_author_counts.most_common(n)]
    logger.info(f"Selected {len(top_co_authors)} co-authors: {top_co_authors}")

    return top_co_authors


def get_author_papers(author: str, j: int, k: int) -> List[arxiv.Result]:
    """
    Get j first-author papers and k non-first-author papers for an author.

    Args:
        author: Author name
        j: Number of first-author papers to select
        k: Number of non-first-author papers to select

    Returns:
        List of arxiv.Result objects (up to j+k papers)
    """
    logger.info(f"Selecting papers for {author}: {j} first-author, {k} non-first-author")

    papers = get_papers_by_author(author)

    # Separate first-author and non-first-author papers
    first_author_papers = []
    non_first_author_papers = []

    for paper in papers:
        if is_first_author(paper, author):
            first_author_papers.append(paper)
        else:
            non_first_author_papers.append(paper)

    # Select j and k papers respectively
    selected_first = first_author_papers[:j]
    selected_non_first = non_first_author_papers[:k]

    logger.info(f"Selected {len(selected_first)} first-author and {len(selected_non_first)} non-first-author papers for {author}")

    return selected_first + selected_non_first


def get_papers(entry: str, n: int, j: int, k: int) -> set[Any]:
    """
    Args:
        entry: Entry point author name
        n: Number of co-authors to include
        j: Number of first-author papers per author
        k: Number of non-first-author papers per author

    Returns:
        Dictionary containing:
        - authors: List of all authors (entry + co-authors)
        - papers: List of deduplicated papers with metadata
        - stats: Authorship statistics per author
    """
    logger.info(f"Generating research summary for {entry} with n={n}, j={j}, k={k}")

    # Step 1: Get co-authors
    co_authors = get_co_authors(entry, n)
    all_authors = [entry] + co_authors

    # Step 2: Process each author
    all_arxiv_ids = set()
    summary_data = []

    for author in all_authors:
        papers = get_author_papers(author, j, k)

        # Deduplicate
        new_papers = [p for p in papers if p.entry_id not in all_arxiv_ids]

        for p in new_papers:
            all_arxiv_ids.add(p.entry_id)

    return all_arxiv_ids


def select_papers_by_author(authors: List[str], n: int = 5) -> Dict[str, List[str]]:
    """
    Given a list of authors, select the `n` most 'relevant' (arxiv metric) papers for each of those.
    Only includes papers where the author is the FIRST author.
    Returns a dictionary mapping author name to a list of paper arXiv IDs.
    """
    logger.info(f"Selecting top {n} first-author papers for {len(authors)} authors...")
    results = {}
    
    client = arxiv.Client()
    
    for author in authors:
        try:
            search = arxiv.Search(
                query=f'au:"{author}"',
                max_results=MAX_RESULTS,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            papers = list(client.results(search))
            paper_ids = []
            
            for p in papers:
                if len(paper_ids) >= n:
                    break
                    
                if is_first_author(p, author) and p.entry_id:
                    paper_ids.append(p.entry_id)
                    
            results[author] = paper_ids
            logger.info(f"Found {len(paper_ids)} first-author papers for {author}")
            
        except Exception as e:
            logger.error(f"Error fetching papers for {author}: {e}")
            results[author] = []
            
    return results

def select_papers_at_least_k_authors(authors: List[str], n: int = 10, k: int = 2) -> List[str]:
    """
    Select the `n` most 'relevant' papers where at least `k` of the authors appear.
    At least one of the `k` authors designated as 'first author'.
    """
    logger.info(f"Selecting top {n} papers where at least {k} authors from list appear...")
    
    client = arxiv.Client()
    paper_map = {} 
    input_authors_norm = {a.lower().replace(" ", "") for a in authors}
    
    for author in authors:
        try:
            search = arxiv.Search(
                    query=f'au:"{author}"',
                    max_results=MAX_RESULTS, 
                    sort_by=arxiv.SortCriterion.Relevance
            )
            
            for p in client.results(search):
                pid = p.entry_id
                
                if pid not in paper_map:
                    paper_map[pid] = {'paper': p, 'overlaps': 0}
                
                paper_map[pid]['overlaps'] += 1
                
        except Exception as e:
             logger.error(f"Error fetching papers for common check {author}: {e}")

    # Iterate and find those with overlaps >= k
    candidates = []
    
    for pid, data in paper_map.items():
        p = data['paper']
        p_authors_norm = {a.name.lower().replace(" ", "") for a in p.authors}
        
        # Intersect
        overlap = len(input_authors_norm.intersection(p_authors_norm))
        
        # Check if first author is in our input list
        first_author_norm = p.authors[0].name.lower().replace(" ", "") if p.authors else ""
        first_author_in_list = first_author_norm in input_authors_norm
        
        if overlap >= k and first_author_in_list:
            candidates.append(p)
    
    # Sort candidates
    candidates.sort(key=lambda x: (len(input_authors_norm.intersection({a.name.lower().replace(" ", "") for a in x.authors})), x.published), reverse=True)
    top_n = candidates[:n]
    
    # Extract entry IDs
    result_ids = []
    for p in top_n:
        if p.entry_id:
            result_ids.append(p.entry_id)
            
    logger.info(f"Found {len(result_ids)} papers meeting commonality criteria.")
    return result_ids


if __name__ == "__main__":
    # Test the new integrated functionality
    all_papers = get_papers(
        entry="Riccardo Salami",
        n=5,  # Get co-authors
        j=5,  # first-author papers per author
        k=5   # non-first-author papers per author
    )

    print(f"Testing with {len(all_papers)} arXiv papers:")
    print(all_papers)