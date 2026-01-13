import arxiv
from typing import List
import os
import csv
from arxiv_handler import search_arxiv, download_source


def normalize_name(name: str) -> str:
    # Very simple normalization: lowercase, collapse spaces
    return " ".join(name.lower().split())


def is_first_author(target_author: str, paper: arxiv.Result) -> bool:
    if not paper.authors:
        return False
    first_author = paper.authors[0].name  # e.g. "John Smith"
    return normalize_name(first_author) == normalize_name(target_author)


def get_papers_by_first_author(
    author_name: str,
    max_results: int = 1000,
    batch_size: int = 200,
) -> List[arxiv.Result]:
    """
    Collect all papers where `author_name` is the *first* author.

    Parameters
    ----------
    author_name : str
        Full name as used on arXiv, e.g. "John Smith".
    max_results : int
        Maximum number of results to examine (API limit safeguard).
    batch_size : int
        Number of results arxiv client is allowed to request in one batch.

    Returns
    -------
    List[arxiv.Result]
        List of arxiv.Result objects matching the first author.
    """
    client = arxiv.Client(page_size=batch_size)

    # arXiv query: all papers where the author appears anywhere in the author list
    # See arXiv API docs for syntax; "au:" filters by author.[web:2][web:6][web:7]
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


def search_author(target_author : str):
    print(f"Searching arXiv for papers with first author: {target_author}")
    papers = get_papers_by_first_author(target_author)

    print(f"Found {len(papers)} papers with {target_author} as first author.\n")

    # Print a simple list; you can also save to CSV, JSON, etc.
    for i, p in enumerate(papers, start=1):
        print(f"[{i}] {p.title}")
        print(f"    arXiv ID: {p.entry_id}")
        print(f"    First author: {p.authors[0].name if p.authors else 'N/A'}")
        print(f"    Published: {p.published}")
        print(f"    PDF: {p.pdf_url}")


def get_all_coauthors(current_author: str, all_authors: set, min_req_papers: int = 3) -> set:
    papers = get_papers_by_first_author(current_author)
    for i, p in enumerate(papers):
        coauthors_list = p.authors
        if coauthors_list:
            for coauthor in coauthors_list:
                if coauthor.name not in all_authors:
                    coauthor_papers = get_papers_by_first_author(coauthor.name)
                    if len(coauthor_papers) >= min_req_papers:
                        print(f"Adding {coauthor.name}")
                        all_authors.add(coauthor.name)
    return all_authors


def collect_author_dict(starting_author: str, max_authors: int = 20) -> dict:

    print('Retrieving all authors...')

    def iterate_authors(all_authors, auth_checked):
        authors_to_check = list(all_authors.copy().difference(auth_checked))
        print(f'Authors to check: {authors_to_check}')
        if (len(authors_to_check) == 0) or (len(list(all_authors)) > max_authors):
            print('Finished searching for authors')
            #return list(all_authors)
        else:
            current_author = authors_to_check[0]
            coauthors = get_all_coauthors(current_author, all_authors)
            all_authors.update(coauthors)
            auth_checked.add(current_author)
            print(f'Found {len(list(all_authors))} authors so far: {list(all_authors)}')
            iterate_authors(all_authors, auth_checked)
            return list(all_authors)

    authors = iterate_authors({starting_author}, set(()))

    if len(authors) == 0:
        print('Error: no authors found')
        return 0
    else:
        print(f'Found a total of {len(authors)} authors:{authors}')
        print('Retrieving all papers for each author...')

        author_dict = {}
        for author in authors:
            papers = get_papers_by_first_author(author)
            author_dict[author] = papers

        return author_dict


def save_author_data(auth_dict: dict, path: str):
    cols = ['arxiv ID', 'title', 'co-authors', 'published', 'category']
    for author in auth_dict.keys():
        try:
            with open(os.path.join(path + author) + ".csv", "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=cols)
                writer.writeheader()
                for p in auth_dict[author]:
                    co_authors = [author.name for author in p.authors]
                    paper_dict = {'arxiv ID': p.entry_id, 'title': p.title, 'co-authors': co_authors, 'published': p.published, 'category': p.categories}
                    writer.writerow(paper_dict)
        except:
            print(f'Could not save author data for {author}')


def get_paper_pool(path, max_authors):
    print("Generating and retrieving pool of papers...")
    author_paper_dict = collect_author_dict("Philipp Hennig", max_authors=max_authors)

    print("Saving author and paper data...")
    save_author_data(author_paper_dict, path)

    return author_paper_dict