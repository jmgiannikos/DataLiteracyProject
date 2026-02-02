import os
import sys
import json
import pandas as pd
import numpy as np
import scipy.stats as sps
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import style_manager as style

style.generate_style()
style.apply_style()

RESULTS = 'data/analysis/author_results_dict.json'
IMGPATH = 'data/analysis/img'
PATH_DATA = 'data/'
def plot_feat_vs_authors(a_r_dict):
    authors = list(a_r_dict.keys())
    a = len(authors)
    features = get_all_features(a_r_dict)
    f = len(features)
    data_arr = np.zeros((f, a))
    for a_i, author in enumerate(authors):
        res_dict = a_r_dict[author]
        for f_i, feature in enumerate(features):
            if feature in list(res_dict.keys()):
                p_val = res_dict[feature][0]
                data_arr[f_i, a_i] = p_val
            else:
                data_arr[f_i, a_i] = 1
    fig = plt.subplots()
    im = plt.imshow(data_arr)
    ax = plt.gca()
    ax.set_xticks(range(a), labels=authors,
                  rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(f), labels=features)

    for i in range(f):
        for j in range(a):
            text = ax.text(j, i, data_arr[i, j],
                           ha="center", va="center", color="w")

    ax.set_title("Title")
    fig.tight_layout()
    plt.show()
    plt.imsave(IMGPATH+'features_vs_authors.png')


def plot_feature_trends(a_r_dict, data, feature):
    authors = list(a_r_dict.keys())
    author_data_dict = {}
    group_data = data
    for author in authors:
        author_data, group_data = split_dataset(author, group_data)
        author_data_dict[author] = author_data
    fig, ax = plt.subplot()
    X_g = group_data['published_x'].to_numpy()
    Y_g = group_data[feature].to_numpy() / group_data['total_words'].to_numpy()
    fig.suptitle(f'Time development of "{feature}"')
    ax.plot(X_g, Y_g, '.', color='black', label='General Data', markersize=0.5)
    for author in authors:
        results = a_r_dict[author]
        author_data = author_data_dict[author]
        _, slope_a, stderr_a, ic_a, stderr_ic_a, _, _, _, _ = results[feature]
        X_a = author_data['published_x'].to_numpy()
        #Y_a = author_data[feature].to_numpy() / author_data['total_words'].to_numpy()
        #ax.plot(X_a, Y_a, 'o', label=f'{author} Data', markersize=6)
        ax.fill_between(X_a, (slope_a-stderr_a) * X_a + ic_a - stderr_ic_a, (slope_a+stderr_a) * X_a + ic_a + stderr_ic_a, alpha=0.2)
        ax.plot(X_a, slope_a * X_a + ic_a, label=f'{author} Model')
    plt.show()
    plt.imsave(IMGPATH + feature + '_trend.png')


def split_dataset(author:str, data:pd.DataFrame) -> [pd.DataFrame, pd.DataFrame]:
    author_set = data[data['first_author'] == author]
    group_set = data[data['first_author'] == author]
    return author_set, group_set

def get_all_features(a_r_dict):
    feature_set = set(())
    for author, results in a_r_dict.items():
        features = list(results.keys())
        feature_set.update(features)
    return list(feature_set)

data = pd.read_csv(PATH_DATA + 'data_time_analysis.csv.gz', index_col='arxiv_id')
with open(RESULTS, 'r') as file:
    auth_res_dict = json.load(file)
#plot_feat_vs_authors(auth_res_dict)

plot_feature_trends(auth_res_dict, data,'need')