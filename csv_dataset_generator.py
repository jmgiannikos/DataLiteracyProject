import enchant
import statistics
import math
from sklearn.linear_model import TheilSenRegressor
import json
import nltk
import os
import csv

def get_sentence_len(sentence):
    words = nltk.tokenize.word_tokenize(sentence, language='english')
    return len(words)

def remove_outlier_sentences(sentences, sentence_lengths, n=1):
    pruned_sentences = []
    pruned_sentence_lenghts = []
    median_len = statistics.median(sentence_lengths)
    stdev_len = statistics.stdev(sentence_lengths)
    for sentence, len in zip(sentences, sentence_lengths):
        if len <= median_len + stdev_len*n and len >= median_len - stdev_len*n:
            pruned_sentences.append(sentence)
            pruned_sentence_lenghts.append(sentence_lengths)
    return pruned_sentences, pruned_sentence_lenghts

# find words, that fall outside of a certain tolerance around what we would expect according to zipfs law
def check_zipfs_law(word_histogram, n = 1):
    remaining_words = {}
    removed_words = {}
    sortable_hist = [(word_histogram[word], word) for word in word_histogram.keys()]
    sortable_hist.sort(reverse=True, key=lambda x: x[0])
    # log scale our word counts, so we can perform outlier detection with a robust linear estimator
    log_scaled_counts = [math.log(x[0], 2) for x in sortable_hist]
    outlier_resistant_estimator = TheilSenRegressor()
    outlier_resistant_estimator.fit(range(len(log_scaled_counts)), log_scaled_counts)
    predicted_counts = outlier_resistant_estimator.predict(range(len(log_scaled_counts)))
    # transfrom back from log space, cutoff linear predictions at zero and round to nearest integer
    expected_counts = [max([int(2**round(pred_count)), 0]) for pred_count in predicted_counts]
    errors = [math.abs(expected_count - sortable_hist[i][0]) for i, expected_count in enumerate(expected_counts)]
    median_error = statistics.median(errors)
    stdev_error = statistics.stdev(errors)
    for i, error in enumerate(errors):
        if error > n*stdev_error + median_error:
            removed_words[sortable_hist[i][1]] = sortable_hist[i][0]
            print(f"removed word: {sortable_hist[i][1]}")
        else:
            remaining_words[sortable_hist[i][1]] = sortable_hist[i][0]
    return remaining_words

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
    return joined_word_hists
            
def transform_to_csv(data_handles, histograms, sentence_lengths):
    col_order = [word for word in histograms[0].keys()]
    rows = []
    for i, hist in enumerate(histograms):
        if i == 0:
            row = ["Data Name"] + col_order + ["Avg Sentence Length", "Stdev Sentence Length"]
        else:
            row = [data_handles[i]] + [hist[word] for word in col_order] + [statistics.mean(sentence_lengths[i]), statistics.stdev(sentence_lengths[i])]
        rows.append(row)
    return rows
    
def generate_csv(sentence_lists, data_handles):
    pruned_word_hists = []
    word_hists = []

    for sentences in sentence_lists:
        sentence_lengths = list(map(get_sentence_len, sentences))
        raw_text = append_sentences(sentences)

        # remove sentences, that deviate more than n*stdev from the median sentence length (default n=1)  
        pruned_sentences, pruned_sentence_lengths = remove_outlier_sentences(sentences, sentence_lengths)

        # get word histogram, excluding the sentences determined as abnormal
        pruned_text = append_sentences(pruned_sentences)
        pruned_words = nltk.tokenize.word_tokenize(pruned_text, language='english')
        pruned_word_hist = check_zipfs_law(get_word_histogram(pruned_words))
        pruned_word_hists.append(pruned_word_hist)

        # get overall word histogram
        words = nltk.tokenize.word_tokenize(raw_text, language='english')
        word_hist = get_word_histogram(words)
        word_hists.append(word_hist)

    csv_union_pruned = transform_to_csv(data_handles, join_word_hists(pruned_word_hists), pruned_sentence_lengths)
    csv_inter_pruned = transform_to_csv(data_handles, join_word_hists(pruned_word_hists, False), pruned_sentence_lengths)

    csv_union_raw = transform_to_csv(data_handles, join_word_hists(word_hists), sentence_lengths)
    csv_inter_raw = transform_to_csv(data_handles, join_word_hists(word_hists, False), sentence_lengths)

    return csv_union_pruned, csv_inter_pruned, csv_union_raw, csv_inter_raw










    