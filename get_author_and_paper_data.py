from arxiv_handler import get_papers_by_first_author


def get_all_coauthors(current_author: str, all_authors: set, excluded_auth, min_req_papers: int = 4) -> tuple[set, set]:
    papers = get_papers_by_first_author(current_author)
    for i, p in enumerate(papers):
        coauthors_list = p.authors
        if coauthors_list:
            for coauthor in coauthors_list:
                name = coauthor.name
                if name not in all_authors and name not in excluded_auth:
                    coauthor_papers = get_papers_by_first_author(name)
                    if len(coauthor_papers) >= min_req_papers:
                        print(f"Adding {name}")
                        all_authors.add(name)
                    else:
                        print(f'Excluding {name}')
                        excluded_auth.add(name)
    return all_authors, excluded_auth


def collect_author_dict(starting_author: str, max_authors: int = 20) -> dict:

    def iterate_authors(all_authors, auth_checked, excluded_auth):
        authors_to_check = list(all_authors.copy().difference(auth_checked))
        print(f'Authors to check: {authors_to_check}')
        if (len(authors_to_check) == 0) or (len(list(all_authors)) > max_authors):
            print('Finished searching for authors')
        else:
            current_author = authors_to_check[0]
            coauthors, excluded_auth = get_all_coauthors(current_author, all_authors, excluded_auth)
            all_authors.update(coauthors)
            auth_checked.add(current_author)
            print(f'Found {len(list(all_authors))} authors so far: {list(all_authors)}')
            iterate_authors(all_authors, auth_checked, excluded_auth)
            return list(all_authors)

    authors = iterate_authors({starting_author}, set(()), set(()))

    if len(authors) == 0:
        print('Error: no authors found')
        return {}
    else:
        print(f'Found a total of {len(authors)} authors: {authors}')
        print('Retrieving all papers for each author...')

        author_dict = {}
        for author in authors:
            papers = get_papers_by_first_author(author)
            author_dict[author] = papers

        return author_dict
