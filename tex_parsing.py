import re
import tarfile
import os
import shutil
from tex_parsing_rules import TEX_PARSING_RULES_LIST
from pylatexenc.latex2text import LatexNodes2Text
import nltk
from utils import import_dataset

def unzip_source_file(source_file_name, staging_dir="./staging"):
    # delete staging folder if it exists, so that we dont mix files from different articles
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)
    os.mkdir(staging_dir)

    with tarfile.open(source_file_name, "r:gz") as tar:
        tar.extractall(path=staging_dir)

    return staging_dir

def get_tex_string(tex_strings):
    tex_string = ""
    for tex_substring in tex_strings:
        tex_string = tex_string + ".".join(tex_substring) # Assume that each file ending also ends any sentences that have not been ended and insert an extra "." to make sure later parsing catches that. Relevant for sentence length stats
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

def process_tex_source(source_file_name, target_file_name, verbose=True):
    # fetching tex source
    if verbose:
        print("Fetching source file...")
    working_folder = unzip_source_file(source_file_name)
    tex_file_contents = import_dataset(working_folder, file_types=["tex"], recursive=True)
    tex_string = get_tex_string(tex_file_contents)
    # clean up temp dir
    shutil.rmtree(working_folder)
    if verbose:
        print("Preprocessing tex string...")
    # apply preprocessing rules defined in TEX_PARSING_RULES_LIST
    preprocessed_tex_string = preprocess_tex_string(tex_string)
    # Initial Latex parsing
    if verbose:
        print("Processing tex string with latexNodes2Text...")
    doc_string = LatexNodes2Text().latex_to_text(preprocessed_tex_string)
    doc_string = postprocess_tex_string(doc_string)
    headings = re.findall("§ .*", doc_string)
    headless_text = re.sub(".*§ .*", "", doc_string)

    with open(target_file_name + ".txt", "w", encoding="utf-8") as txtfile:
        txtfile.write(doc_string)

    # get sentences and sentence, lengths
    sentences = nltk.tokenize.sent_tokenize(headless_text, language='english')

    return sentences, headings
