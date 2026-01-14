import arxiv
from utils import realize_path, strip_entry_id, sanitize_article_id
from arxiv_handler import download_source
from tex_parsing import process_tex_source


def process_article(article: arxiv.Result, author: str) -> tuple[dict, list]:
    """
    :param article:
    :param author:
    :return:
    - returns dictionary with article data and sentence list
    - also downloads .tex and saves to folder
    """
    try:
        zip_target_path = f"./tex_sources/{author}"
        realize_path(zip_target_path, overwrite=False)
        paper_arxiv_id = article.entry_id
        article_id = strip_entry_id(sanitize_article_id(paper_arxiv_id))
        zip_path = download_source(strip_entry_id(paper_arxiv_id), zip_target_path)

        try:
            print("Getting Document String...")

            processed_txt_path = f"./processed_tex_sources/{author}/"
            realize_path(processed_txt_path, overwrite=False)
            sentences, headings = process_tex_source(zip_path, processed_txt_path, article_id)
            article_data_dict = create_paper_data_dict(article)
            print(f"Processing article with ID {paper_arxiv_id} successful!")

        except Exception as inst:
            print(type(inst))
            print(inst)
            print(f"Processing article with ID {paper_arxiv_id} failed")
            return {}, []

    except Exception as inst:
        print(type(inst))
        print(inst)
        print(f"Retrieving article by {author} failed")
        return {}, []

    return article_data_dict, sentences


def create_paper_data_dict(paper: arxiv.Result):
    first_author = paper.authors[0].name
    co_authors = [author.name for author in paper.authors[1:]]
    paper_dict = {'arxiv ID': paper.entry_id, 'first author': first_author, 'co-authors': co_authors,
                  'published': paper.published, 'category': paper.categories}
    return paper_dict
