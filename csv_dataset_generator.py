import enchant
import statistics
import math
from sklearn.linear_model import TheilSenRegressor
import json
import nltk
import os
import csv
import numpy as np
from nltk.stem import PorterStemmer

def get_sentence_len(sentence):
    words = nltk.tokenize.word_tokenize(sentence, language='english')
    return len(words)

def remove_outlier_sentences(sentences, sentence_lengths, n=4):
    if len(sentence_lengths) > 2:
        pruned_sentences = []
        pruned_sentence_lenghts = []
        mean_len = statistics.mean(sentence_lengths)
        stdev_len = statistics.stdev(sentence_lengths)
        for sentence, length in zip(sentences, sentence_lengths):
            if length <= mean_len + stdev_len*n and length >= mean_len - stdev_len*n:
                pruned_sentences.append(sentence)
                pruned_sentence_lenghts.append(length)
        return pruned_sentences, pruned_sentence_lenghts
    else:
        return sentences, sentence_lengths

# find words, that fall outside of a certain tolerance around what we would expect according to zipfs law
# NOTE: weirdly empty word histograms appear here  
def check_zipfs_law(word_histogram, n = 4, verbose=True):
    remaining_words = {}
    removed_words = {}
    sortable_hist = [(word_histogram[word], word) for word in word_histogram.keys()]
    sortable_hist.sort(reverse=True, key=lambda x: x[0])
    # according to zips law the inverted fraction should be roughly linear, proportional to rank
    inverted_counts = [x[0]**(-1) for x in sortable_hist]

    print(len(inverted_counts))
    if len(inverted_counts) <= 1:
        return word_histogram

    outlier_resistant_estimator = TheilSenRegressor()
    index_array = np.expand_dims(np.arange(len(inverted_counts), step=1), axis=1)
    outlier_resistant_estimator.fit(index_array, inverted_counts)
    predicted_counts = outlier_resistant_estimator.predict(index_array)
    print(predicted_counts.shape)

    # transform back to expected counts and round to integer
    expected_counts = [max([int(round(pred_count**-1)), 0]) if round(pred_count) > 0 else 0 for pred_count in predicted_counts]

    # compute errors
    # errors are "normalized" by the actual count. an error of 10 if the word was counted 100 times is less anomalous than the same error on a count of 2
    errors = [(expected_count - sortable_hist[i][0])/sortable_hist[i][0] for i, expected_count in enumerate(expected_counts)]

    # compute stdev and median (since median  is more outlier resistant)
    median_error = statistics.median(errors)
    stdev_error = statistics.stdev(errors)

    for i, error in enumerate(errors):
        # if error falls outside n stdev band around median remove word
        if error > median_error + n*stdev_error or error < median_error - n*stdev_error:
            removed_words[sortable_hist[i][1]] = sortable_hist[i][0]
            if verbose:
                print(f"removed word: {sortable_hist[i][1]}")
        else:
            remaining_words[sortable_hist[i][1]] = sortable_hist[i][0]
    return remaining_words

# NOTE: stemming may be a good way to reduce amount of words to get more actionable datasets
# by default we apply it to the pruned dataset, but not to the raw one
def get_word_histogram(text_words, stemm=False):
    if stemm:
        porter_stemmer = PorterStemmer()
    d = enchant.Dict("en_US")
    word_hist = {}
    for word in text_words:
        # prune all non english words
        if d.check(word) and not (len(word) == 1 and not (word == "a" or word =="i")):
            if stemm:
                word = porter_stemmer.stem(word)
            if word in word_hist.keys():
                word_hist[word] += 1
            else:
                word_hist[word] = 1
    return word_hist

def append_sentences(sentences):
    text = ""
    for sentence in sentences:
        text += sentence + " "
    return text

# unify the word histograms, so they are constructed over the same set of words
# can be done in union mode (default), where the dictionaries are filled with entries of frequency zero
# or it can be run in intersection mode, where only words that are contained in every document are retained
def join_word_hists(word_hists, union=True):
    # get word list
    lexicon = []
    if union:
        for word_hist in word_hists:
            contained_words = list(word_hist.keys())
            # add unique words to word list (lexicon)
            lexicon = list(set(lexicon + contained_words))
    else:
        for i, word_hist in enumerate(word_hists):
            contained_words = list(word_hist.keys())
            if i == 0:
                lexicon = contained_words
            else:
                # calculate intersection between word list and lexicon
                lexicon = list(set(lexicon) & set(contained_words))

    # expand/reduce word hists
    joined_word_hists = []
    for word_hist in word_hists:
        joined_word_hist = {}
        for word in lexicon:
            if word in word_hist.keys():
                joined_word_hist[word] = word_hist[word]
            else:
                joined_word_hist[word] = 0
        joined_word_hists.append(joined_word_hist)

    words = joined_word_hists[0].keys()
    sum_dict = {}
    for word_hist in joined_word_hists:
        if word_hist.keys() != words:
            print("## Word misalignment between supposedly joinable word hists ##")
        for word in word_hist.keys():
            if word in sum_dict.keys():
                sum_dict[word] += word_hist[word]
            else:
                sum_dict[word] = word_hist[word]
    for word in sum_dict.keys():
        if sum_dict[word] <= 0:
            print("## issue in join ##")
            print(f"word {word} has zero occurences")

    return joined_word_hists
            
def transform_to_csv(data_handles, histograms, sentence_lengths):
    col_order = [word for word in histograms[0].keys()]
    rows = []
    for i, hist in enumerate(histograms):
        if i == 0:
            row = ["Data Name"] + col_order
        else:
            row = [data_handles[i]] + [hist[word] for word in col_order]
        rows.append(row)
    return rows

def generate_sentence_json(sentence_lists, data_handles):
    pruned_json_dict = {}
    raw_json_dict = {}
    for i, sentences in enumerate(sentence_lists):
        sentence_lenths = list(map(get_sentence_len, sentences))
        raw_json_dict[data_handles[i]] = sentence_lenths
        _, pruned_sentence_lenths = remove_outlier_sentences(sentences, sentence_lenths)
        pruned_json_dict[data_handles[i]] = pruned_sentence_lenths
    return pruned_json_dict, raw_json_dict
    
def generate_wordhist_csv(sentence_lists, data_handles):
    pruned_word_hists = []
    word_hists = []

    pruned_sentence_lengths_list = []
    sentence_lengths_list = []

    for i, sentences in enumerate(sentence_lists):
        sentence_lengths = list(map(get_sentence_len, sentences))

        raw_text = append_sentences(sentences)

        # remove sentences, that deviate more than n*stdev from the median sentence length (default n=4)  
        pruned_sentences, pruned_sentence_lengths = remove_outlier_sentences(sentences, sentence_lengths)
        pruned_sentence_lengths_list.append(pruned_sentence_lengths)

        # get word histogram, excluding the sentences determined as abnormal
        pruned_text = append_sentences(pruned_sentences)
        pruned_words = nltk.tokenize.word_tokenize(pruned_text, language='english')
        pruned_word_hist = check_zipfs_law(get_word_histogram(pruned_words, stemm=True))
        pruned_word_hists.append(pruned_word_hist)

        # get overall word histogram
        words = nltk.tokenize.word_tokenize(raw_text, language='english')
        word_hist = get_word_histogram(words)
        for word in word_hist.keys():
            if word_hist[word] == 0:
                print("## Issue in word hist generation ##")
                print(f"word {word} has count zero")
        for word in pruned_word_hist.keys():
            if word_hist[word] == 0:
                print("## Issue in pruned word hist generation ##")
                print(f"word {word} has count zero")
        word_hists.append(word_hist)


    csv_union_pruned = transform_to_csv(data_handles, join_word_hists(pruned_word_hists), pruned_sentence_lengths_list)
    csv_inter_pruned = transform_to_csv(data_handles, join_word_hists(pruned_word_hists, False), pruned_sentence_lengths_list)

    csv_union_raw = transform_to_csv(data_handles, join_word_hists(word_hists), sentence_lengths_list)
    csv_inter_raw = transform_to_csv(data_handles, join_word_hists(word_hists, False), sentence_lengths_list)

    return csv_union_pruned, csv_inter_pruned, csv_union_raw, csv_inter_raw










    