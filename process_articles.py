from utils import realize_path, strip_entry_id, sanitize_article_id
from arxiv_handler import download_source, search_arxiv
from tex_parsing import process_tex_source


def process_articles(articles, author, verbose=True):
    i = 0
    article_handles = []
    sentence_lists = []

    for article in articles:
        try:
            zip_target_path = f"./tex_sources/{author}"
            realize_path(zip_target_path, overwrite=False)
            paper_arxiv_id = article.entry_id
            article_id = strip_entry_id(sanitize_article_id(paper_arxiv_id))
            zip_path = download_source(strip_entry_id(paper_arxiv_id), zip_target_path)

            try:
                if verbose:
                    print("Getting Document String...")

                processed_txt_path = f"./processed_tex_sources/{author}/"
                realize_path(processed_txt_path, overwrite=False)
                sentences, headings = process_tex_source(zip_path, processed_txt_path, article_id, verbose=verbose)

                # collect sentence outputs and handles
                sentence_lists.append(sentences)
                article_handles.append(f"{author}/{article_id}")

                if verbose:
                    print(f"Processing article {i} with ID {paper_arxiv_id} successful!")

            except Exception as inst:
                if verbose:
                    print(type(inst))
                    print(inst)
                    print(f"Processing article {i} with ID {paper_arxiv_id} failed")

        except Exception as inst:
            if verbose:
                print(type(inst))
                print(inst)
                print(f"Retrieving article {i} by {author} failed")
        i += 1
    return sentence_lists, article_handles


