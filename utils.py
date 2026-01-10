import os
import shutil
import json

def sanitize_article_id(article_id):
    return article_id.replace(".", "_")

def strip_entry_id(arxiv_entry_id):
    return arxiv_entry_id.split("/")[-1]

def realize_path(path, overwrite=True):
    splitpath = path.split("/")
    last_subpath = ""
    for i, path_section in enumerate(splitpath):
        if i == 0:
            current_subpath = path_section
        else:
            last_subpath = current_subpath
            current_subpath = last_subpath + "/" + path_section

        if os.path.isdir(current_subpath):
            # optionally overwrites the leaf directory (and its included data) by deleting it and creating a new dir with the same name
            if overwrite and i == len(splitpath)-1:
                shutil.rmtree(current_subpath)
                os.mkdir(current_subpath)
        else:
            os.mkdir(current_subpath)

# currently allows for recursive call, but assumes that even in a multi dir environment the
# individual file handles are unique. TODO: fix this so duplicate data handles cannot occur.
def import_dataset(dataset_location, file_types=[], recursive=False):
    data_dir_contents = os.listdir(dataset_location)
    data_contents = []
    data_handles = []
    for element in data_dir_contents:
        element_path = dataset_location + "/" + element
        if os.path.isdir(element_path) and recursive:
            local_data, local_data_handles = import_dataset(element_path, file_types, recursive=True)
            data_contents = data_contents + local_data
            data_handles = data_handles + local_data_handles
        elif not os.path.isdir(element_path):
            file_type = element.split(".")[-1]
            if file_type in file_types:
                if file_type == "json":
                    with open(element_path, 'r') as file:
                        data_content = json.load(file)
                elif file_type == ".tex":
                    with open(element_path, "r", encoding="utf-8") as texfile:
                        data_content = texfile.read()
                else:
                    with open(element_path, "r", encoding="utf-8") as file:
                        data_content = file.read()

                data_contents.append(data_content)
                data_handles.append(element.split(".")[0])
    return data_contents, data_handles


def import_from_txt(path):
    author_paper_dict = {}
    author_file_list = os.listdir(path)
    for author_file in author_file_list:
        author_file_loc = os.path.join(path, author_file)
        papers = open(author_file_loc).read().splitlines()
        author = os.path.basename(author_file)[:-4]
        paper_dict_list = []
        for paper in papers:
            paper_dict = {'title': paper.split(';')[0], 'authors': paper.split(';')[1].split(',')}
            paper_dict_list = paper_dict_list + [paper_dict]
        author_paper_dict[author] = paper_dict_list
    return author_paper_dict
