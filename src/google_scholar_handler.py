"""
Google Scholar HTML parsing utilities.
Parses exported Google Scholar author pages to extract paper information.

Source: jan-analysis/google_scholar_handler.py
"""

import re
from typing import List, Dict, Optional

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False


def remove_tags_from_all(strings: List[str]) -> List[str]:
    """
    Remove HTML tags from a list of strings.

    Args:
        strings: List of strings potentially containing HTML tags

    Returns:
        List of strings with HTML tags removed
    """
    return [re.sub(r"<.*?>", "", s) for s in strings]


def parse_papers_page(
    html_doc: str,
    author: Optional[str] = None
) -> List[Dict]:
    """
    Parse Google Scholar papers page HTML.

    Extracts paper titles and author lists from Google Scholar author page exports.
    Optionally filters to only include papers where the specified author is first author.

    Args:
        html_doc: Raw HTML string from Google Scholar
        author: Optional author name to filter first-author papers

    Returns:
        List of paper dictionaries with 'title' and 'authors' keys

    Source: jan-analysis/google_scholar_handler.py
    """
    if not HAS_BS4:
        raise ImportError("BeautifulSoup4 is required for Google Scholar parsing. "
                         "Install with: pip install beautifulsoup4")

    soup = BeautifulSoup(html_doc, 'html.parser')
    article_elements = soup.findAll(class_="gs_mnde_one_art")

    articles = []
    for article_element in article_elements:
        lines = list(article_element.children)

        if len(lines) < 2:
            continue

        # Extract title from first line
        title_elem = lines[0].contents
        title = title_elem[0] if title_elem else ""

        # Extract authors from second line
        # Use stripped_strings because the searched name may be wrapped in <b> tags
        author_string = " ".join(list(lines[1].stripped_strings))
        authors = remove_tags_from_all(author_string.split(","))
        authors = [a.strip() for a in authors if a.strip()]

        # Filter by first author if specified
        if author is not None:
            if not authors or author != authors[0]:
                continue

        article = {
            "title": title,
            "authors": authors
        }
        articles.append(article)

    return articles


def load_and_parse_scholar_html(
    file_path: str,
    author: Optional[str] = None
) -> List[Dict]:
    """
    Load and parse a Google Scholar HTML file.

    Args:
        file_path: Path to HTML file
        author: Optional author name to filter first-author papers

    Returns:
        List of paper dictionaries
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    return parse_papers_page(html_content, author)


def extract_papers_from_directory(
    directory_path: str,
    filter_first_author: bool = True
) -> Dict[str, List[Dict]]:
    """
    Extract papers from all HTML files in a directory.

    Assumes HTML files are named after the author (e.g., "John_Smith.html").

    Args:
        directory_path: Path to directory containing HTML files
        filter_first_author: If True, only include papers where file author is first author

    Returns:
        Dictionary mapping author names to their paper lists
    """
    import os
    from pathlib import Path

    results = {}
    html_files = Path(directory_path).glob("*.html")

    for html_file in html_files:
        author_name = html_file.stem.replace("_", " ")
        filter_author = author_name if filter_first_author else None

        try:
            papers = load_and_parse_scholar_html(str(html_file), filter_author)
            results[author_name] = papers
        except Exception as e:
            print(f"Error parsing {html_file}: {e}")
            results[author_name] = []

    return results
