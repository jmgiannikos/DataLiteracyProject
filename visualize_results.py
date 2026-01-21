import matplotlib.pyplot as plt
import statistics
import os
import json

def visualize_word_histogram(word_histogram):
    dict_list = [(word_histogram[word], word) for word in word_histogram.keys()]
    dict_list.sort(key=lambda x: x[0], reverse=True)
    show_list = dict_list[:20]
    total_num = sum([tup[0] for tup in dict_list])
    words = [tup[1] for tup in show_list]
    freq = [tup[0]/total_num for tup in show_list]

    plt.bar(words, freq)
    plt.title('top 20 word frequencies')
    plt.xlabel('Word')
    plt.ylabel('Frequency')
    plt.show(block=True)

def visualize_sentence_len_histogram(sentence_lens):
    plt.hist(sentence_lens, bins=100)
    plt.xlabel('Sentence Length')
    plt.ylabel('Frequency')
    plt.show()

def visualize_dict_results(result_dict):
    word_histogram = result_dict["word_hist"]
    visualize_word_histogram(word_histogram)
    sentence_lengths = result_dict["sentence_lengths"]
    clean_sentence_lens = [length for length in sentence_lengths if length < 100]
    visualize_sentence_len_histogram(clean_sentence_lens)
    mean_sentence_lenght = sum(clean_sentence_lens)/len(clean_sentence_lens)
    std_sentence_lenght = statistics.stdev(clean_sentence_lens)
    return mean_sentence_lenght, std_sentence_lenght

def visualize_results(result_dict_dir):
    files = os.listdir(result_dict_dir)
    mean_sent_lens = []
    std_sent_lens = []
    for file in files:
        try:
            with open(result_dict_dir + "/" + file, "r") as f:
                result = json.load(f)
            avg_lens, std_lens = visualize_dict_results(result)
            mean_sent_lens.append(avg_lens)
            std_sent_lens.append(std_lens)
        except:
            print("failure")



    doc_labels = range(len(mean_sent_lens))
    plt.bar(doc_labels, mean_sent_lens, yerr=std_sent_lens)
    plt.title('avg sentence lens per doc')
    plt.xlabel('doc id')
    plt.ylabel('avg sentence len')
    plt.show()

visualize_results("/home/jan-malte/Desktop/DataLiteracyProject/processed_tex_sources")
