import re
import tarfile
import os
import shutil
from tex_parsing_rules import TEX_PARSING_RULES_LIST
from pylatexenc.latex2text import LatexNodes2Text
import json
import nltk

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

def process_tex_source(source_file_name, target_file_name):
    # fetching tex source
    working_folder = unzip_source_file(source_file_name)
    tex_file_paths = find_all_tex_sources(working_folder)
    tex_string = get_tex_string(tex_file_paths)

    # apply preprocessing rules defined in TEX_PARSING_RULES_LIST
    preprocessed_tex_string = preprocess_tex_string(tex_string)
    
    # Initial Latex parsing
    doc_string = LatexNodes2Text().latex_to_text(preprocessed_tex_string)
    doc_string = postprocess_tex_string(doc_string)
    headings = re.findall("§ .*", doc_string)
    headless_text = re.sub(".*§ .*", "", doc_string)

    # get sentences and sentence, lengths
    sentences = nltk.tokenize.sent_tokenize(headless_text, language='english')

    with open(target_file_name, "w") as outfile:
        outfile.write(json.dumps(sentences))

    return sentences