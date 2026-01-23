# Pipeline initiator file
# select authors -> select papers -> scrape metadata -> scrape text -> clean text -> extract features
# visualization will be done in a separate jupyter notebook

from src.scrape_paper_ids import get_papers
from src.scrape_metadata import scrape_metadata_arxivIDs
import src.scrape_text as scr
import src.extract_features as extract

# Physics Author
arxiv_ids_florentin = get_papers(
    entry="Florentin Millour",
    n=100,  # Get co-authors
    j=10000,  # first-author papers per author
    k=100   # non-first-author papers per author
)

# Biology Author
arxiv_ids_manel = get_papers(entry="Manel Gil-Sorribes",
    n=100,  # Get co-authors
    j=10000,  # first-author papers per author
    k=100   # non-first-author papers per author
)

# Econ Author
arxiv_ids_wen = get_papers(entry="Wen Lou",
    n=100,  # Get co-authors
    j=10000,  # first-author papers per author
    k=100   # non-first-author papers per author
)

arxiv_ids = arxiv_ids_florentin.union(arxiv_ids_manel).union(arxiv_ids_wen)

print(f"Testing with {len(arxiv_ids)} arXiv papers:")
for arxiv_id in arxiv_ids:
    print(f"  - {arxiv_id}")
print()

metadata_dataframe = scrape_metadata_arxivIDs(arxiv_ids)

print(f"Processed {len(metadata_dataframe)} papers")
print(f"\nDOIs found: {metadata_dataframe['doi'].notna().sum()}/{len(metadata_dataframe)}")
print(f"\nMetadata sources:")
print(metadata_dataframe['metadata_source'].value_counts())
print(f"\nAverage completeness: {metadata_dataframe['metadata_completeness'].mean():.2%}")
print(f"\nOutput saved to: data/metadata.csv")

scr.main()
extract.main()



