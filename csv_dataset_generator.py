import enchant
import statistics
import math
from sklearn.linear_model import TheilSenRegressor
import json
import nltk

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

def import_dataset(dataset_location):
    pass


def generate_csv(dataset_location):
    sentence_lists = import_dataset(dataset_location)

    for sentences in sentence_lists:
        sentence_lengths = list(map(get_sentence_len, sentences))
        raw_text = append_sentences(sentences)

        # remove sentences, that deviate more than n*stdev from the median sentence length (default n=1)  
        pruned_sentences, pruned_sentence_lengths = remove_outlier_sentences(sentences, sentence_lengths)

        # get word histogram, excluding the sentences determined as abnormal
        pruned_text = append_sentences(pruned_sentences)
        pruned_words = nltk.tokenize.word_tokenize(pruned_text, language='english')
        pruned_word_hist = check_zipfs_law(get_word_histogram(pruned_words))

        # get overall word histogram
        words = nltk.tokenize.word_tokenize(raw_text, language='english')
        word_hist = get_word_histogram(words)