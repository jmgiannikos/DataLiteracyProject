import urllib
from utils import sanitize_article_id, normalize_name
import time
from typing import List
import arxiv


def get_papers_by_first_author(
    author_name: str,
    max_results: int = 200,
    batch_size: int = 4,
) -> List[arxiv.Result]:
    """
    Collect all papers where `author_name` is the *first* author.

    Parameters
    ----------
    author_name : str
        Full name as used on arXiv
    max_results : int
        Maximum number of results to examine (API limit safeguard).
    batch_size : int
        Number of results arxiv client is allowed to request in one batch.

    Returns
    -------
    List[arxiv.Result]
        List of arxiv.Result objects matching the first author.
    """
    time.sleep(4)
    client = arxiv.Client(page_size=batch_size)

    query = f'au:"{author_name}"'

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    first_author_papers: List[arxiv.Result] = []

    for result in client.results(search):
        if is_first_author(author_name, result):
            first_author_papers.append(result)

    return first_author_papers


def download_source(article_id, target_path):
    time.sleep(5)
    download_link = f"https://export.arxiv.org/e-print/{article_id}"
    urllib.request.urlretrieve(download_link, f"{target_path}/{sanitize_article_id(article_id)}")
    return f"{target_path}/{sanitize_article_id(article_id)}"


def is_first_author(target_author: str, paper: arxiv.Result) -> bool:
    if not paper.authors:
        return False
    first_author = paper.authors[0].name  # e.g. "John Smith"
    return normalize_name(first_author) == normalize_name(target_author)