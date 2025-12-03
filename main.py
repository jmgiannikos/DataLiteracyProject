from google_scholar_handler import parse_papers_page
from arxiv_handler import search_arxiv, download_source
from utils import strip_entry_id, sanitize_article_id, import_dataset, realize_path
from tex_parsing import process_tex_source
from csv_dataset_generator import generate_csv
import csv

HTML_DOC_PATH = "/home/jan-malte/DataLiteracyProject/AuthorPages"
CSV_FILE_PATHS= ["./union_pruned.csv", "./inter_pruned.csv", "./union_raw.csv", "./inter_raw.csv"] # NOTE: must have exactly 4 values, that are valid file paths

def process_articles(articles, author_handle, verbose=True):
    i = 0
    e = 0
    article_handles = []
    sentence_lists = []
    for article in articles:
        try:
            if verbose:
                print(f"Searching arxiv for article {i}...")
            arxiv_search_result = search_arxiv(article)
            if verbose:
                print("Getting ID...")
            paper_arxiv_id = arxiv_search_result.entry_id
            if verbose:
                print("Downloading...")

            zip_target_path = f"./tex_sources/{author_handle}"
            realize_path(zip_target_path, overwrite=False)
            zip_path = download_source(strip_entry_id(paper_arxiv_id), zip_target_path)

            try:
                if verbose:
                    print("Getting Document String...")

                processed_txt_path = f"./processed_tex_sources/{author_handle}/"
                realize_path(processed_txt_path, overwrite=False)
                sentences, headings = process_tex_source(zip_path, processed_txt_path, verbose=verbose)

                # collect sentence outputs and handles
                sentence_lists.append(sentences)
                article_handles.append(f"{author_handle}/{strip_entry_id(sanitize_article_id(paper_arxiv_id))}")
                
                if verbose:
                    print(f"Processing article {i} with ID {paper_arxiv_id} successful!")
            except Exception as inst:
                if verbose:
                    print(type(inst))
                    print(inst)
                    print(f"Processing article {i} with ID {paper_arxiv_id} failed")
                e += 1
        except:
            if verbose:
                print(f"Fetching article {i} failed")
            e += 1
        i += 1
    if verbose:
        print(f"Processing of {e} out of {i} articles failed")
    return sentence_lists, article_handles


def main():
    html_paper_pages, author_handles = import_dataset(HTML_DOC_PATH, file_types=["html"])
    global_sentence_lists = [] 
    global_article_handles = []
    
    for i, html_page in enumerate(html_paper_pages):
        articles = parse_papers_page(html_page)
        sentence_lists, article_handles = process_articles(articles, author_handles[i])
        global_sentence_lists = global_sentence_lists + sentence_lists
        global_article_handles = global_article_handles + article_handles


    print(global_article_handles)
    i = 0
    for csv_variant in generate_csv(global_sentence_lists, global_article_handles):
        with open(CSV_FILE_PATHS[i], 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(csv_variant)
        i += 1

if __name__ == "__main__":
    main()
