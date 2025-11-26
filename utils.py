def sanitize_article_id(article_id):
    return article_id.replace(".", "_")

def strip_entry_id(arxiv_entry_id):
    return arxiv_entry_id.split("/")[-1]