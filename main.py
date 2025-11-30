from google_scholar_handler import read_papers_page, parse_papers_page
from arxiv_handler import search_arxiv, download_source
from utils import strip_entry_id, sanitize_article_id
from tex_parsing import process_tex_source

HTML_DOC_PATH = "/home/jan-malte/DataLiteracyProject/AuthorPages/Henning.html"

def process_articles(articles):
    i = 0
    for article in articles:
        try:
            arxiv_search_result = search_arxiv(article)
            paper_arxiv_id = arxiv_search_result.entry_id
            zip_path = download_source(strip_entry_id(paper_arxiv_id))
            doc_string = process_tex_source(zip_path, f"./processed_tex_sources/{strip_entry_id(sanitize_article_id(paper_arxiv_id))}")
        except:
            print(f"parsing article {i} failed")
        i += 1

def main():
    html_document = read_papers_page(HTML_DOC_PATH)
    articles = parse_papers_page(html_document)
    process_articles(articles)

if __name__ == "__main__":
    main()