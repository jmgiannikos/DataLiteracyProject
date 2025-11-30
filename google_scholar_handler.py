from bs4 import BeautifulSoup
import re

def remove_tags_from_all(strings):
    sanitized_list = []
    for string in strings:
        sanitized_list.append(re.sub("<.*>", "", string))
    return sanitized_list

def read_papers_page(html_doc_path):
    with open(html_doc_path, "r", encoding="utf-8") as file:
        html_content = file.read()
    return html_content

def parse_papers_page(html_doc, author=None):
    soup = BeautifulSoup(html_doc, 'html.parser')
    article_elements = soup.findAll(class_="gs_mnde_one_art") # fetch the article elements by css class
    articles = []
    for atricle_element in article_elements:
        lines = list(atricle_element.children)
        title = lines[0].contents
        author_string = " ".join(list(lines[1].stripped_strings)) # because the searched name is encapsulated by <b> we cant simply use contents here
        authors = remove_tags_from_all(author_string.split(","))
        
        # if an author name is provided, filter all publications where author in question is not first author
        if (author is not None and author == authors[0]) or author is None:
            article = {
                "title": title[0], # title comes wrapped in a one element list due to contents
                "authors": authors
            }
            articles.append(article)

    return articles