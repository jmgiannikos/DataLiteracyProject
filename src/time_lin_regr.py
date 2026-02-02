import pandas as pd
import numpy as np
import scipy.stats as sps
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import random as rn
from data_utils import remove_papers_with_sen_length
import time
import os
import sys
import json

PATH_DATA = 'data/'
PATH_METADATA = 'data/data/cleaned-with-features.csv'
PATH_PUBLISHED_DATE = 'data/data/enriched-arxiv-dataset.csv'
PATH_WORD_HIST_UNION = 'data/data/features/word_histogram_union_raw.csv'
AUTHORS = ['Hongjie Dong', 'A. Mironov', 'Holger Dette', 'Duván Cardona', 'Liping Li', 'T. Tony Cai', 'Gen Li', 'László Csató']


def prepare_data(save_csv=True, path=PATH_DATA):
    print('Loading Metadata')
    metadata_df = pd.read_csv(PATH_METADATA, delimiter=',', index_col='arxiv_id', usecols=['arxiv_id','first_author', 'total_words', 'mean_sentence_length', 'vocabulary_size'])
    date_df = pd.read_csv(PATH_PUBLISHED_DATE, delimiter=',', usecols=['arxiv_id','published'], index_col='arxiv_id')
    metadata = pd.merge(metadata_df, date_df, how='inner', left_index=True, right_index=True)
    print(f'Metadata columns: {metadata.columns.tolist()}\n'
          f'Retained columns correctly: {len(metadata_df.columns.tolist()) + len(date_df.columns.tolist()) == len(metadata.columns.tolist())}\n'
          f'Number of Samples: {len(metadata.index.tolist())}')
    print('Cleaning Dataframe')
    metadata = remove_papers_with_sen_length(metadata, 5, 300)
    print('Converting Time')
    metadata['published'] = metadata['published'].apply(convert_time)
    print('Loading Word Histogram')
    raw_union_df = import_word_hist('data/normalized_word_hist', PATH_WORD_HIST_UNION, min_count=10000, min_words=50,
                                    time_proc=True)
    print('Merging datasets')
    data = pd.merge(metadata, raw_union_df, how='inner', left_index=True, right_index=True)
    print(f'Retained columns correctly: {len(metadata.columns.tolist()) + len(raw_union_df.columns.tolist()) == len(data.columns.tolist())}')
    print('Deleting intermediate dataframes')
    del raw_union_df, metadata_df, date_df, metadata
    print(f'Final Dataset size: {len(data)}')
    if save_csv:
        print('Saving Data to '+ path)
        data.to_csv(path + 'data_time_analysis.csv.gz', header=True, index=True, compression='gzip')
    return data

def import_word_hist(target_path, path=PATH_WORD_HIST_UNION, chunksize=300, min_count=5000, min_words=50, time_proc=False):
    if os.path.exists(target_path):
        sys.exit(f'Warning: {target_path} file already exists')
    print('Reading Columns (Features) of Word Hist')
    header_df = pd.read_csv(path, nrows=0, delimiter=',', dtype=int, index_col='Data Name', na_filter=False, converters={'Data Name':float})
    cols = header_df.columns.tolist()
    print('Setting up Count Dataframe')
    count_df = pd.DataFrame(index=['count'],columns=cols)
    i = 0
    header_df.to_csv(target_path + '_full.csv', mode='a', columns=cols, index=True, index_label='arxiv_id', header=True)
    with pd.read_csv(path, chunksize=chunksize, delimiter=',', dtype=int, index_col='Data Name', na_filter=False, converters={'Data Name':float}) as reader:
        for chunk in reader:
            if time_proc:
                start = time.time()
            print(f'Processing Chunk {i}')
            df = chunk[chunk.sum(axis=1) > min_words]
            print('Saving Chunk')
            df.to_csv(target_path + '_full.csv', mode='a', columns=cols, index=True, header=False)
            count_df['count'] += df.sum(axis=0)
            del df
            if time_proc:
                print(f'Time for Chunk {i}: {time.time()-start}')
                del start
            i += 1
    print('Reducing columns')
    cols = count_df[count_df['count'] > min_count].columns.tolist()
    del count_df
    word_hist_df = pd.read_csv(target_path + '_full.csv', dtype=int, converters={'arxiv_id':float}, na_filter=False, index_col='arxiv_id', usecols=['arxiv_id'] + cols)
    word_hist_df.to_csv(target_path + '.csv', mode='w', columns=cols, index=True, index_label='arxiv_id', header=True)
    return word_hist_df

def split_dataset(author:str, data:pd.DataFrame) -> [pd.DataFrame, pd.DataFrame]:
    author_set = data[data['first_author'] == author]
    group_set = data[data['first_author'] != author]
    return author_set, group_set

def find_sign_features_for_author(author:str, presel_features:list, data:pd.DataFrame):
    author_data, group_data = split_dataset(author, data)
    author_features_df = find_best_features(feature_list=presel_features, author_data=author_data, n=20)
    print(f'Author feature dataframe: {author_features_df.head(15)}')
    selected_features = author_features_df.index.tolist()
    print(f'Selected features for {author}: {selected_features}')
    general_features_df = fit_group_data(feature_lst=selected_features, group_data=group_data)
    print(f'Group feature dataframe: {general_features_df.head(15)}')
    significant_features = test_features(author_features_df, general_features_df, selected_features)
    print(f'Significant features for {author}: {significant_features}')
    return significant_features

def convert_time(date):
    date = date[:10]
    year = int(date[:4])
    month = int(date[5:7])
    day = int(date[-2:])
    result = year + (month - 1) / 12 + (day - 1) / 365
    return result

def find_best_features(feature_list: list, author_data: pd.DataFrame, n:int=20)-> pd.DataFrame:
    feat_fit_df = pd.DataFrame(index=feature_list, columns=['slope', 'intercept', 'rvalue', 'pvalue', 'stderr', 'stderr_intercept'])
    if len(author_data.index.tolist()) == 0:
        sys.exit("Author Dataframe is empty")
    X = author_data['published_x'].to_numpy()
    for feature in feature_list:
        Y = author_data[feature].to_numpy()/author_data['total_words'].to_numpy()
        result = sps.linregress(X, Y)
        feat_fit_df.loc[feature] = [result.slope, result.intercept, result.rvalue, result.pvalue, result.stderr, result.intercept_stderr]
    feat_fit_df = feat_fit_df.sort_values(by=['stderr'], ascending=True)
    return feat_fit_df.head(n)

def fit_group_data(feature_lst: list, group_data: pd.DataFrame)->pd.DataFrame:
    fit_group_df = pd.DataFrame(index=feature_lst, columns=['slope', 'intercept', 'rvalue', 'pvalue', 'stderr', 'stderr_intercept'])
    for feature in feature_lst:
        X = group_data['published_x'].to_numpy()
        Y = group_data[feature].to_numpy()/group_data['total_words'].to_numpy()
        result = sps.linregress(X, Y)
        fit_group_df.loc[feature] = [result.slope, result.intercept, result.rvalue, result.pvalue, result.stderr, result.intercept_stderr]
    return fit_group_df

def test_statistic(beta1:float, beta2:float, s1:float, s2:float)->float:
    t = (beta1 - beta2)/np.sqrt(s1**2 - s2**2)
    return t

def test_features(author_fits:pd.DataFrame, group_fits:pd.DataFrame, features:list, p=0.05)->dict:
    result_dict = {}
    for feature in features:
        slope_author, interc_a, _, _, stderr_author, stderr_ic_author = author_fits.loc[feature].tolist()
        slope_group, interc_g, _, _, stderr_group, stderr_ic_group = group_fits.loc[feature].tolist()
        t_stat = test_statistic(slope_author, slope_group, stderr_author, stderr_group)
        p_val = sps.norm.sf(t_stat)
        print(f'Test results for {feature}: \n t = {t_stat} \n p-value = {p_val}')
        if p_val < p:
            result_dict[feature] = (p_val, slope_author, stderr_author, interc_a, stderr_ic_author, slope_group, stderr_group, interc_g, stderr_ic_group)
    return result_dict

def plot_all(auth_res_dict:dict, data:pd.DataFrame):
    authors = list(auth_res_dict.keys())
    rn.seed(67)
    colors = rn.sample(list(mcolors.XKCD_COLORS.values()), len(authors))
    # Here, we assign each top author a fixed color
    author_color_dict = {author: color for author, color in zip(authors, colors)}
    for author, results in auth_res_dict.items():
        plot_color = author_color_dict[author]
        author_data, group_data = split_dataset(author, data)
        features = list(results.keys())
        n_feat = min(len(features), 5)
        if n_feat == 0:
            print(f'No features for {author}')
            continue
        fig, ax = plt.subplot(n_feat)
        fig.suptitle(author)
        i = 0
        for feature in features:
            if i >= n_feat:
                continue
            _, slope_a, _, ic_a, _, slope_g, _, ic_g, _= results[feature]
            X_a = author_data['published_x'].to_numpy()
            Y_a = author_data[feature].to_numpy()/author_data['total_words'].to_numpy()
            X_g = group_data['published_x'].to_numpy()
            Y_g = group_data[feature].to_numpy()/group_data['total_words'].to_numpy()
            ax[i].plot(X_g, Y_g, '.', color='black', label='General Data', markersize=1)
            ax[i].plot(X_g, slope_g * X_g + ic_g, color='black', label='General Model')
            ax[i].plot(X_a, Y_a, 'o', color=plot_color, label=f'{author} Data', markersize=6)
            ax[i].plot(X_a, slope_a * X_a + ic_a, color=plot_color, label=f'{author} Model')
            ax[i].set_title(feature)
            i+=1

def time_analysis(prep_data=True, authors=AUTHORS, visualize=True):
    if prep_data:
        data = prepare_data()
        print(f'Prepared data for {len(data)} papers with {len(data.columns.tolist())-1} features')
    else:
        data = pd.read_csv(PATH_DATA + 'data_time_analysis.csv.gz', index_col='arxiv_id')
        print(f'Loaded data for {len(data)} papers with {len(data.columns.tolist()) - 1} features')
    pres_features = data.columns.tolist()[5:]
    print(f'Preselected {len(pres_features)} features')
    author_results_dict = {}
    for author in authors:
        print(f'Processing for {author}')
        auth_res = find_sign_features_for_author(author, pres_features, data)
        author_results_dict[author] = auth_res
        print(auth_res)
    if visualize:
        plot_all(author_results_dict, data)
    # Serialize data into file:
    json.dump(author_results_dict, open("data/analysis/author_results_dict.json", 'w'))

time_analysis(prep_data=True, visualize=False)
