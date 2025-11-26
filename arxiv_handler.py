import arxiv
import urllib
from utils import sanitize_article_id

def search_arxiv(paper, client=arxiv.Client()):
    search = arxiv.Search(
        query=paper["title"],
        max_results=1,
        sort_by=arxiv.SortCriterion.Relevance
    )
    return list(client.results(search))[0]

def download_source(article_id):
    download_link = f"https://arxiv.org/src/{article_id}"
    urllib.request.urlretrieve(download_link, f"tex_sources/{sanitize_article_id(article_id)}")
    return f"./tex_sources/{sanitize_article_id(article_id)}"