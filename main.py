from get_author_and_paper_data import collect_author_dict
from retrieve_article_data import process_article
import csv


def main():
    # Path for saving the data of papers
    path = "C:/Users/rapha/Documents/Uni/Master_Med_Strahlenwissenschaften/Med_Str-wissenschaften_S3/DataLiteracy" \
           "/Project/Project-Raphael/authors_and_papers/"

    # Searches for first authors and corresponding papers based on initial author and stops when number of
    # authors after an iteration is reached (meaning there will be more than max_authors)
    print("Generating and retrieving pool of papers...")
    author_dict = collect_author_dict("Paul Norbury", max_authors=10)

    # downloads .tex sources and processes them to obtain data
    process_author_dict(author_dict, path)


def process_author_dict(author_dict, path):
    cols = ['arxiv ID', 'first author', 'co-authors', 'published', 'category']
    with open(path + "paper_data.csv", "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=cols)
        writer.writeheader()
        for author in author_dict.keys():
            articles = author_dict[author]
            for article in articles:
                data, sentences = process_article(article, author)
                writer.writerow(data)
                # TODO: Creation of word and sentence length histograms for each paper


if __name__ == "__main__":
    main()
