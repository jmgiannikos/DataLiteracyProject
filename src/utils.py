"""
Utility functions for the data literacy project.
Consolidated from origin/retrieve-data and origin/jan-analysis branches.
"""

import os
import shutil
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import re
import numpy as np


def realize_path(path: str, overwrite: bool = True) -> None:
    """
    Create directory path recursively.

    Args:
        path: Directory path to create
        overwrite: If True, removes and recreates the leaf directory if it exists

    Source: retrieve-data/utils.py, jan-analysis/utils.py
    """
    splitpath = path.split("/")
    last_subpath = ""
    for i, path_section in enumerate(splitpath):
        if i == 0:
            current_subpath = path_section
        else:
            last_subpath = current_subpath
            current_subpath = last_subpath + "/" + path_section

        if os.path.isdir(current_subpath):
            # Optionally overwrite the leaf directory
            if overwrite and i == len(splitpath) - 1:
                shutil.rmtree(current_subpath)
                os.mkdir(current_subpath)
        else:
            os.mkdir(current_subpath)


def import_dataset(
    dataset_location: str,
    file_types: Optional[List[str]] = None,
    recursive: bool = False
) -> Tuple[List, List[str]]:
    """
    Import files from a directory with optional type filtering and recursion.

    Args:
        dataset_location: Path to the directory containing files
        file_types: List of file extensions to include (e.g., ['json', 'txt', 'tex'])
        recursive: If True, search subdirectories recursively

    Returns:
        Tuple of (data_contents, data_handles) where handles are filenames without extension

    Source: retrieve-data/utils.py, jan-analysis/utils.py
    """
    if file_types is None:
        file_types = []

    data_dir_contents = os.listdir(dataset_location)
    data_contents = []
    data_handles = []

    for element in data_dir_contents:
        element_path = os.path.join(dataset_location, element)

        if os.path.isdir(element_path) and recursive:
            local_data, local_data_handles = import_dataset(element_path, file_types, recursive=True)
            data_contents.extend(local_data)
            data_handles.extend(local_data_handles)
        elif not os.path.isdir(element_path):
            file_type = element.split(".")[-1]
            if file_type in file_types:
                if file_type == "json":
                    with open(element_path, 'r', encoding='utf-8') as file:
                        data_content = json.load(file)
                elif file_type == "tex":
                    with open(element_path, "r", encoding="utf-8") as texfile:
                        data_content = texfile.read()
                else:
                    with open(element_path, "r", encoding="utf-8") as file:
                        data_content = file.read()

                data_contents.append(data_content)
                data_handles.append(element.rsplit(".", 1)[0])

    return data_contents, data_handles


def sanitize_article_id(article_id: str) -> str:
    """
    Convert arXiv ID or DOI to a safe filename.

    Replaces special characters (/, :, .) with underscores.

    Args:
        article_id: arXiv ID or DOI string

    Returns:
        Sanitized string safe for use as filename

    Source: retrieve-data/utils.py, scrape_metadata.py
    """
    return article_id.replace("/", "_").replace(":", "_").replace(".", "_")


def strip_entry_id(arxiv_entry_id: str) -> str:
    """
    Extract article ID from full arXiv entry URL.

    Args:
        arxiv_entry_id: Full arXiv URL (e.g., 'http://arxiv.org/abs/2103.00020')

    Returns:
        Just the article ID (e.g., '2103.00020')

    Source: retrieve-data/utils.py, jan-analysis/utils.py
    """
    return arxiv_entry_id.split("/")[-1]


def import_from_txt(path: str) -> Dict[str, List[Dict]]:
    """
    Import author-paper mappings from text files.

    Expects files in format: title;author1,author2,...
    One paper per line.

    Args:
        path: Directory containing author text files (named {author}.txt)

    Returns:
        Dictionary mapping author names to lists of paper dicts with 'title' and 'authors' keys

    Source: jan-analysis/utils.py
    """
    author_paper_dict = {}
    author_file_list = os.listdir(path)

    for author_file in author_file_list:
        if not author_file.endswith('.txt'):
            continue

        author_file_loc = os.path.join(path, author_file)
        papers = open(author_file_loc, encoding='utf-8').read().splitlines()
        author = os.path.basename(author_file)[:-4]  # Remove .txt extension

        paper_dict_list = []
        for paper in papers:
            parts = paper.split(';')
            if len(parts) >= 2:
                paper_dict = {
                    'title': parts[0],
                    'authors': [a.strip() for a in parts[1].split(',')]
                }
                paper_dict_list.append(paper_dict)

        author_paper_dict[author] = paper_dict_list

    return author_paper_dict


def ensure_dir(path: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure exists
    """
    Path(path).mkdir(parents=True, exist_ok=True)

def merge_duplicate_authors(metadata_df, author_col = "first_author"):
    authors = metadata_df[author_col]
    author_unifier = [author.replace(".", "").replace("-", " ").lower() for author in authors]
    metadata_df.drop(author_col, axis = 1, inplace = True)
    metadata_df[author_col] = author_unifier
    return metadata_df

def remove_duplicate_papers(metadata_df, paper_name_col="title"):
    retained_papers = []
    retained_papers_map = []
    for paper_name in metadata_df[paper_name_col]:
        standardized_paper_name = re.sub(r'[^\w\s]', '', paper_name.lower())
        if not standardized_paper_name in retained_papers:
            retained_papers.append(standardized_paper_name)
            retained_papers_map.append(True)
        else:
            retained_papers_map.append(False)
    metadata_df = metadata_df[retained_papers_map]
    return metadata_df

def select_top_n_authors(metadata_df, author_name_col="first_author", n=6):
    authors = list(set(metadata_df[author_name_col].to_list()))
    author_maps = np.zeros((metadata_df.shape[0], len(authors)))
    map_idx = 0
    for _, row in metadata_df.iterrows():
        author = row [author_name_col]
        author_idx = authors.index(author)
        author_maps[map_idx][author_idx] = 1
        map_idx += 1
    author_paper_counts = np.sum(author_maps, axis=0)
    topn_author_idxs =  np.argpartition(author_paper_counts, -n)[-n:]
    author_maps = author_maps[:,topn_author_idxs]
    topn_author_map = np.sum(author_maps, axis=1)
    return metadata_df[topn_author_map != 0]


