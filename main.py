from process_articles import process_articles
from get_author_and_paper_data import get_paper_pool


def main():
    global_sentence_lists = []
    global_article_handles = []

    path = "C:/Users/rapha/Documents/Uni/Master_Med_Strahlenwissenschaften/Med_Str-wissenschaften_S3/DataLiteracy/Project/Project-Raphael/authors_and_papers/"
    author_dict = get_paper_pool(path, max_authors=10)
    for author in author_dict.keys():
        articles = author_dict[author]
        sentence_lists, article_handles = process_articles(articles, author)
        global_sentence_lists = global_sentence_lists + sentence_lists
        global_article_handles = global_article_handles + article_handles


if __name__ == "__main__":
    main()
