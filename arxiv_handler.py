import arxiv
import urllib
from utils import sanitize_article_id
import os

def search_arxiv(paper, client=arxiv.Client()):
    search = arxiv.Search(
        query=paper["title"],
        max_results=1,
        sort_by=arxiv.SortCriterion.Relevance
    )
    return list(client.results(search))[0]

def download_source(article_id, target_path):
    download_link = f"https://arxiv.org/src/{article_id}"
    target_path_full = f"{target_path}/{sanitize_article_id(article_id)}"
    if not os.path.isfile(target_path_full): # Why are we doing this?
        urllib.request.urlretrieve(download_link, target_path_full)
    else:
        print("ERROR: path invalid")
    return f"{target_path}/{sanitize_article_id(article_id)}"