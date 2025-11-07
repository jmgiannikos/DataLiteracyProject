from bs4 import BeautifulSoup
import re
import arxiv
import urllib
import tarfile
import os
import shutil
from pylatexenc.latex2text import LatexNodes2Text
from tex_parsing_rules import TEX_PARSING_RULES_LIST
import nltk
import enchant
import json

def remove_tags_from_all(strings):
    sanitized_list = []
    for string in strings:
        sanitized_list.append(re.sub("<.*>", "", string))
    return sanitized_list

def parse_papers_page(html_doc):
    soup = BeautifulSoup(html_doc, 'html.parser')
    article_elements = soup.findAll(class_="gs_mnde_one_art") # fetch the article elements by css class
    articles = []
    for atricle_element in article_elements:
        lines = list(atricle_element.children)
        title = lines[0].contents
        author_string = " ".join(list(lines[1].stripped_strings)) # because the searched name is encapsulated by <b> we cant simply use contents here
        authors = remove_tags_from_all(author_string.split(","))
        article = {
            "title": title[0], # title comes wrapped in a one element list due to contents
            "authors": authors
        }
        articles.append(article)
    return articles

def search_arxiv(paper, client=arxiv.Client()):
    search = arxiv.Search(
        query=paper["title"],
        max_results=1,
        sort_by=arxiv.SortCriterion.Relevance
    )
    return list(client.results(search))[0]

def read_papers_page(html_doc_path):
    with open(html_doc_path, "r", encoding="utf-8") as file:
        html_content = file.read()
    return html_content

def download_source(article_id):
    download_link = f"https://arxiv.org/src/{article_id}"
    urllib.request.urlretrieve(download_link, f"tex_sources/{sanitize_article_id(article_id)}")
    return f"./tex_sources/{sanitize_article_id(article_id)}"

def sanitize_article_id(article_id):
    return article_id.replace(".", "_")

def strip_entry_id(arxiv_entry_id):
    return arxiv_entry_id.split("/")[-1]

def unzip_source_file(source_file_name, staging_dir="./staging"):
    # delete staging folder if it exists, so that we dont mix files from different articles
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)
    os.mkdir(staging_dir)

    with tarfile.open(source_file_name, "r:gz") as tar:
        tar.extractall(path=staging_dir)

    return staging_dir

def find_all_tex_sources(directory_path):
    tex_file_paths = []
    dir_contents = os.listdir(directory_path)
    for element in dir_contents:
        element_path = directory_path + "/" + element
        if os.path.isdir(element_path):
            local_tex_file_paths = find_all_tex_sources(element_path)
            tex_file_paths = tex_file_paths + local_tex_file_paths
        else:
            if ".tex" in element:
                tex_file_paths.append(element_path)
    return tex_file_paths

def get_tex_string(file_paths):
    tex_string = ""
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as texfile:
            tex_substring = texfile.read()
        tex_string = tex_string + "." + tex_substring # Assume that each file ending also ends any sentences that have not been ended and insert an extra "." to make sure later parsing catches that. Relevant for sentence length stats
    return tex_string

def preprocess_tex_string(tex_string):
    for rule in TEX_PARSING_RULES_LIST:
        tex_string = rule(tex_string)
    return tex_string

def postprocess_tex_string(tex_string):
    tex_string = re.sub(r"< g r a p h i c s >", "", tex_string)
    tex_string = re.sub("\n{2,}\s*", "\n", tex_string) # collapse long chains of newlines
    tex_string = re.sub(r"\\n{2,}\s*", "\n", tex_string)
    tex_string = tex_string.lower()
    return tex_string

def get_sentences(string):
    string = re.sub(".*§ .*", "", string)
    sentences =  nltk.tokenize.sent_tokenize(string, language='english')
    return sentences

def get_word_histogram(text_words):
    d = enchant.Dict("en_US")
    word_hist = {}
    for word in text_words:
        if d.check(word) and not (len(word) == 1 and not (word == "a" or word =="i")):
            if word in word_hist.keys():
                word_hist[word] += 1
            else:
                word_hist[word] = 1
    return word_hist

def get_sentence_len(sentence):
    words = nltk.tokenize.word_tokenize(sentence, language='english')
    return len(words)

def process_tex_source(source_file_name, target_file_name):
    working_folder = unzip_source_file(source_file_name)
    tex_file_paths = find_all_tex_sources(working_folder)
    tex_string = get_tex_string(tex_file_paths)
    preprocessed_tex_string = preprocess_tex_string(tex_string)
    doc_string = LatexNodes2Text().latex_to_text(preprocessed_tex_string)
    doc_string = postprocess_tex_string(doc_string)
    headings = re.findall("§ .*", doc_string)
    headless_text = re.sub(".*§ .*", "", doc_string)
    sentences = nltk.tokenize.sent_tokenize(headless_text, language='english')
    sentence_lengths = list(map(get_sentence_len, sentences))
    words = nltk.tokenize.word_tokenize(headless_text, language='english')
    word_hist = get_word_histogram(words)
    result_dict = {
        "word_hist": word_hist,
        "sentence_lengths": sentence_lengths,
        "headings": headings
    }
    with open(target_file_name, "w") as outfile:
        outfile.write(json.dumps(result_dict))

    return result_dict

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

HTML_DOC_PATH = "/home/jan-malte/Desktop/DataLiteracyProject/AuthorPages/Henning.html"
html_document = read_papers_page(HTML_DOC_PATH)
articles = parse_papers_page(html_document)
process_articles(articles)