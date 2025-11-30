from google_scholar_handler import read_papers_page, parse_papers_page
from arxiv_handler import search_arxiv, download_source
from utils import strip_entry_id, sanitize_article_id
from tex_parsing import process_tex_source

HTML_DOC_PATH = "/home/jan-malte/DataLiteracyProject/AuthorPages/Henning.html"

def process_articles(articles):
    i = 0
    e = 0
    for article in articles:
        try:
            print(f"Searching arxiv for article {i}...")
            arxiv_search_result = search_arxiv(article)
            print("Getting ID...")
            paper_arxiv_id = arxiv_search_result.entry_id
            print("Downloading...")
            zip_path = download_source(strip_entry_id(paper_arxiv_id), "tex_sources")
            try:
                print("Getting Document String...")
                doc_string = process_tex_source(zip_path, f"./processed_tex_sources/{strip_entry_id(sanitize_article_id(paper_arxiv_id))}")
                print(f"Processing article {i} with ID {paper_arxiv_id} successful!")
            except Exception as inst:
                print(type(inst))
                print(inst)
                print(f"Processing article {i} with ID {paper_arxiv_id} failed")
                e += 1
        except:
            print(f"Fetching article {i} failed")
            e += 1
        i += 1
    print(f"Processing of {e} out of {i} articles failed")


def main():
    html_document = read_papers_page(HTML_DOC_PATH)
    articles = parse_papers_page(html_document)
    process_articles(articles)

if __name__ == "__main__":
    main()
