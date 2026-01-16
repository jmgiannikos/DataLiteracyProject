# DataLiteracyProject
## Usage
- The Author Pages folder contains the html of the article export page of a given author (example: https://scholar.google.com/citations?view_op=list_mandates_page_export&hl=en&user=MUg_qAYAAAAJ). This has to be collected manually currently, as google scholar blocks automated requests.
- playfground.py file contains a brief pipeline, that fetches all articles for a given author page. Note: before that works, one should make sure the folders "processed_tex_sources/" and "tex_sources/" exist in the main project folder.
- to see some example visualisations run visualize_results.py after running playfground.py
  
## main.py

Will be the pipeline orchestrator. 

select authors -> select papers -> scrape metadata -> scrape text -> clean text -> extract features

Visualization will be done in a separate jupyter notebook.

## select_papers.py

Handles arXiv queries to scrape a list of papers in the form of arXiv IDs.

All methods are helpers except for `get_papers()`.

```python
"""
Args:
        entry: Entry point author name
        n: Number of co-authors to include
        j: Number of first-author papers per author
        k: Number of non-first-author papers per author
        strict: If True, require exactly n co-authors, j first-author papers, and k non-first-author papers.
                If False (default), allow up to n, j, and k respectively.

    Returns:
        Set of arXiv IDs for all deduplicated papers
"""
```

TODO:
- [ ] recursive call to build a larger set of papers
- [ ] include "at least `m` co-authors among the `n` selected should appear in `i` papers among the `j+k` selected for each author"

## scrape_metadata.py

Main method: `run()`. First, it extracts basic metadata from arXiv. If DOI present, searches via OpenAlex (highest hitrate possible). If not, tries CrossRef.
Right now, 50% of papers have full metadata representation.

```python
"""
    Takes a list of arXiv IDs, fetches DOIs from arXiv API, then enriches with
    metadata from OpenAlex/Crossref.

    Args:
        arxiv_ids: List of arXiv identifiers
        contact_email: Contact email for polite API access (or set CONTACT_EMAIL env var)
        cache_dir: Base directory for caching API responses
        output_path: Path for output metadata.csv

    Returns:
        DataFrame with enriched metadata
    """
```

## scrape_text.py

Sim